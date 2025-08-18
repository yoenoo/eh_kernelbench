import json
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from typing import Optional
from transformers.trainer_utils import get_last_checkpoint

import uuid
import shutil
from pathlib import Path

import sys, os, re
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(repo_root)
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.random_words import generate_random_word
from src.kernelbench_eval.run_parallel import parallel_eval_lists
from src.kernelbench_eval.utils import set_gpu_arch
set_gpu_arch(["Hopper"])

SYSTEM_PROMPT = ""

# add near imports
import tempfile, uuid, traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import torch.multiprocessing as mp

# import wandb

# use_wandb = True
# if use_wandb:
#   wandb.init(project="grpo-kernel-bench", name="grpo-kernel-bench")

#     print(run.name)
#     print(run.id)


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


def reward_correctness(completions, **kwargs):
  original_src_dir = Path("./original_src")
  original_src_dir.mkdir(exist_ok=True, parents=True)
  target_src_dir = Path("./target_src")
  target_src_dir.mkdir(exist_ok=True, parents=True)

  originals = []
  targets = []
  for idx, (c, original_src) in enumerate(zip(completions, kwargs["code"])):
    text = c[0]["content"]
    target_src = extract_kernel(text)

    original_src_path = original_src_dir / f"{uuid.uuid4().hex}.py"
    target_src_path = target_src_dir / f"{uuid.uuid4().hex}.py"

    original_src_path.write_text(original_src.strip())
    target_src_path.write_text(target_src.strip() if target_src is not None else "")

    originals.append(original_src_path)
    targets.append(target_src_path)

  results = parallel_eval_lists(originals, targets, max_gpus=4, runs=10, seed=42, print_progress=True)
  rewards = []
  for res in results:
    if res.is_correct:
      reward = 0.3 + res.median_speed_up
      reward = min(reward, 10.0) ## clamp to keep GRPO stable
    elif res.is_executed:
      reward = 0.05
    elif res.is_compiled:
      reward = 0.01
    else:
      reward = -1.0

    rewards.append(reward)

  # # cleanup
  # shutil.rmtree(original_src_dir)
  # shutil.rmtree(target_src_dir)

  # original_src_dir.rmdir(force=True)
  # target_src_dir.rmdir(force=True)

  print(rewards)
  return rewards


import time
from pathlib import Path

def get_shared_output_dir(base="grpo-qwen3-14b-checkpoints"):
    marker = Path(".shared_output_dir.txt")
    rank = int(os.environ.get("RANK", "0"))

    if rank == 0:
        # If resuming, reuse existing name; else create a new one
        if marker.exists():
            out = marker.read_text().strip()
        else:
            out = f"{base}-{generate_random_word()}"
            marker.write_text(out)
    else:
        # Wait until rank 0 writes the name
        while not marker.exists():
            time.sleep(0.05)
        out = marker.read_text().strip()

    Path(out).mkdir(parents=True, exist_ok=True)
    return out

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
    target_modules=[
      "q_proj","k_proj","v_proj","o_proj",
      "gate_proj","up_proj","down_proj"
    ],
  )

  # random_word = generate_random_word()
  # output_dir = f"grpo-qwen3-14b-checkpoints-{random_word}"
  # Path(output_dir).mkdir(exist_ok=True, parents=True)

  output_dir = get_shared_output_dir("grpo-qwen3-14b-checkpoints")

  training_args = GRPOConfig(
    output_dir=output_dir,
    
    temperature=1.0,
    top_p=0.95,
    learning_rate = 1e-4,
    # adam_beta1 = 0.9,
    # adam_beta2 = 0.99,
    # weight_decay = 0.1,
    # warmup_ratio = 0.1,
    # lr_scheduler_type = "cosine",

    # optim="paged_adamw_8bit",
    
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

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
    max_steps=500,
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
    logging_strategy="steps",
    logging_steps=1,
    logging_first_step=True,
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