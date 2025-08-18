from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from typing import Optional

import sys, os, re
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from transformers.trainer_utils import get_last_checkpoint

SYSTEM_PROMPT = ""

# add near imports
import tempfile, uuid, traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import torch.multiprocessing as mp

# import wandb

# use_wandb = True
# if use_wandb:
#   with wandb.init(project="grpo-kernel-bench", name="grpo-kernel-bench") as run:
#     print(run.name)
#     print(run.id)


# one-time, at module import
mp.set_start_method("spawn", force=True)

# Optional: limit per-rank pool to 1 worker to serialize evals on that rank
_EVAL_POOL = None
def _get_pool():
    global _EVAL_POOL
    if _EVAL_POOL is None:
        _EVAL_POOL = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
    return _EVAL_POOL

def _safe_eval_worker(original_src: str, candidate_src: str, device_id: int, atol: float, rtol: float, timeout_s: int = 120):
    # This runs in a separate process → separate CUDA context
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # Inside this subprocess, the visible device is index 0
    import torch
    torch.cuda.set_device(0)
    from eval import run_eval_core

    # optional: unique temp file for debugging/repro
    import tempfile, uuid, pathlib
    tmpdir = tempfile.mkdtemp(prefix=f"kernel_eval_{uuid.uuid4().hex[:8]}_")
    kernel_path = str(pathlib.Path(tmpdir) / "kernel.py")
    with open(kernel_path, "w") as f:
        f.write(candidate_src)

    # IMPORTANT: use device_id=0 inside the worker (mapped by CUDA_VISIBLE_DEVICES)
    return run_eval_core(
        original_src=original_src,
        candidate_src=candidate_src,
        device_id=0,
        atol=atol,
        rtol=rtol,
        verbose=False,
    )

def reward_correctness(completions, **kwargs):
    rewards = []
    # Use the actual current device of this DDP process
    try:
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
    except Exception:
        device_id = int(os.environ.get("LOCAL_RANK", "0"))

    # print(f"{len(completions)} completions..!", device_id)

    pool = _get_pool()

    for idx, (c, original_src) in enumerate(zip(completions, kwargs["code"])):
        text = c[0]["content"]
        # print(f"{len(rewards)} tries..")

        candidate_src = extract_kernel(text)
        if candidate_src is None:
            rewards.append(-1.0)  # no code block → penalize
            continue

        # Never call CUDA in the parent on failure; run eval in a sandboxed subprocess
        fut = pool.submit(_safe_eval_worker, original_src, candidate_src, device_id, 1e-2, 1e-2)

        try:
            runtime_ref, runtime_new = fut.result(timeout=180)  # adjust as needed
            # Robust reward shaping
            if runtime_new <= 0 or not (runtime_new < float("inf")):
                reward = 0.0
            else:
                reward = 0.3 + (runtime_ref / runtime_new)
        except TimeoutError:
            # hung kernel – kill the worker process tree and respawn on next call
            reward = 0.0
        except Exception:
            # kernel crashed (illegal access, etc.)
            # DO NOT call torch.cuda.* here; just return a safe reward
            # You can log for debugging:
            # traceback.print_exc()
            reward = 0.0

        rewards.append(float(max(min(reward, 10.0), -1.0)))  # clamp to keep GRPO stable

    print(rewards)
    return rewards


def build_dataset(name: str, split: str, limit: Optional[int] = None):
  ds = load_dataset(name, split=split)
  if limit:
    ds = ds.select(range(min(limit, len(ds))))
  ds = ds.map(lambda x: {"prompt": prompt_generate_custom_cuda_from_prompt_template(x["code"])})
  return ds

_CODE_REGEX = r"```python\s*(.*?)```"
_CODE_FENCE = re.compile(_CODE_REGEX, re.DOTALL | re.IGNORECASE)
# def reward_format_box(completions, **kwargs):
#   rewards = []
#   for c in completions:
#     text = c[0]["content"]
#     if re.search(_CODE_REGEX, text, re.DOTALL | re.MULTILINE):
#       rewards.append(1)
#     else:
#       rewards.append(0)
#   return rewards

def extract_kernel(text: str) -> Optional[str]:
  matches = _CODE_FENCE.findall(text)
  if not matches:
    return None
  code = matches[-1].strip().strip("`").strip()
  return code

# def reward_correctness(completions, **kwargs):
#   rewards = []
#   for c, original_src in zip(completions, kwargs["code"]):
#     text = c[0]["content"]
#     # print(text)
#     print(f"{len(rewards)} tries..")

#     # step 1: parse out kernel solution from the completion
#     candidate_src = extract_kernel(text)

#     # step 2: save the kernel solution to a file 
#     if candidate_src is not None:
#       with open("kernel.py", "w") as f:
#         f.write(candidate_src)
#     else:
#       rewards.append(-1.0)  # penalize for not following the format box
#       continue

#     # step 3: execute the saved kernel file and check compiled/correctness/runtime
#     from eval import run_eval_core
#     try:
#       device_id = int(os.environ.get("LOCAL_RANK", "0"))
#       runtime_ref, runtime_new = run_eval_core(original_src=original_src, candidate_src=candidate_src, device_id=device_id, atol=1e-2, rtol=1e-2, verbose=False)
#     except:
#       import torch, gc
#       if torch.cuda.is_available():
#         torch.cuda.synchronize()
#         torch.cuda.empty_cache()
#         gc.collect()

#       rewards.append(0.0)
#       continue

#     # step 4: based on the result from step 3, give a reward
#     reward = 0.3 + runtime_ref / runtime_new
#     rewards.append(reward)

#   return rewards


if __name__ == "__main__":

  dataset = build_dataset("ScalingIntelligence/KernelBench", split="level_1")
  dataset = dataset.map(lambda x: {
    "prompt" : [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user",   "content": x["prompt"] + " /think"},
    ]
  })

  peft_cfg = LoraConfig(
    r=16, 
    lora_alpha=32, 
    lora_dropout=0, 
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
  )

  training_args = GRPOConfig(
    output_dir="grpo-qwen3-14b-checkpoints-brisk",
    
    bf16=True,
    optim="paged_adamw_8bit",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    use_vllm=True,
    # vllm_mode="colocate", # this leads to OOM in my setting (8 x H200)
    vllm_mode="server",
    vllm_server_base_url="http://127.0.0.1:8000",
    vllm_gpu_memory_utilization=0.8,
    
    ## TODO: check if below speeds up training
    generation_batch_size=64,   # try 64 → 96 → 128
    # vllm_server_timeout=1800,
    
    num_generations=8,
    max_prompt_length=1024 + 512,
    # max_completion_length=100,
    max_completion_length=8192,
    logging_steps = 1,
    model_init_kwargs={
      "torch_dtype": "bfloat16",
      "use_cache": False,  # saves activation memory in training
    },

    # multi-gpu stuff
    ddp_find_unused_parameters=True,
    ddp_broadcast_buffers=False,        # avoids extra sync on buffers
    remove_unused_columns=False,        # HF Trainer won’t drop inputs you use
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # safer with PyTorch 2.x

    # logging
    save_steps=5,
    # report_to = "wandb" if use_wandb else "none",
  )

  trainer = GRPOTrainer(
    # model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    model="Qwen/Qwen3-14B",
    args=training_args,
    reward_funcs=[reward_correctness],
    train_dataset=dataset,
    peft_config=peft_cfg,
  )

  ckpt = get_last_checkpoint(training_args.output_dir)
  if ckpt:
    with open(os.path.join(ckpt, "trainer_state.json")) as f:
      st = json.load(f)
    print("Resuming from:", ckpt, "global_step:", st.get("global_step"))
    trainer.train(resume_from_checkpoint=ckpt)
  else:
    print("No checkpoint found, starting fresh.")
    trainer.train()