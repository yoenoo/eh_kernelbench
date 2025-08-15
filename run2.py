import os 
import re
import torch
import asyncio
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

from src.utils import save_kernel
from src.vllm_backend import run_vllm, init_engine
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template

from dotenv import load_dotenv
load_dotenv()

KERNEL_SAVE_DIR = Path("results/")
KERNEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

## TODO: tidy up
if __name__ == "__main__":

  from huggingface_hub import login
  access_token=os.environ["HF_TOKEN"]
  login(token=access_token)

  base_model = "Qwen/QwQ-32B" ## TODO: automate
  model_name = "cognition-ai/Kevin-32B"

  tokenizer = AutoTokenizer.from_pretrained(model_name)

  dataset_name = "ScalingIntelligence/KernelBench"  
  dataset = load_dataset(dataset_name, split="level_1")
  dataset = dataset.map(lambda x: {"prompt": prompt_generate_custom_cuda_from_prompt_template(x["code"])})

  # auto-configure tensor parallelism for multiple GPUs
  num_gpus = torch.cuda.device_count()
  tp_size = num_gpus if num_gpus and num_gpus > 1 else 1

  # Initialize engine with base model
  engine = init_engine(
    model_path=base_model, 
    dtype="bfloat16", 
    gpu_memory_utilization=0.95,
    max_model_len=32768,
    tensor_parallel_size=tp_size
    # max_num_batched_tokens=4096,                     
    # max_num_seqs=64,                               # cap concurrent sequences inside vLLM
    # kv_cache_dtype="fp8",                          # <- enable on H200 if quality ok
    # speculative_model="Qwen/Qwen2.5-7B-Instruct",  # <- drafter
    # num_speculative_tokens=6,                      # 4–8 is a good sweep
  )

  def parse_fn(example, completions):
    level = example["level"]
    problem_id = example["problem_id"]
    for sample_id, c in enumerate(completions):
      try:
        custom_kernel = re.findall(r"```python(.*)```", c, re.DOTALL)[-1].strip()
        fname = f"kernel_level_{level}_problem_{problem_id}_sample_{sample_id}.py"
        save_kernel(KERNEL_SAVE_DIR / fname, custom_kernel)
      except IndexError:
        print(f"no kernel found for level {level}, problem {problem_id}, sample {sample_id}")

  args = dict(
    system_prompt="", 
    n_samples=16, 
    max_tokens=16384, 
    # max_tokens=16,
    use_think=True, 
    parse_fn=parse_fn,
    temperature=1.0,
    top_p=0.95,
  )

  asyncio.run(run_vllm(engine, tokenizer, dataset, **args))