from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.vllm_backend import run_vllm, init_engine
from datasets import load_dataset
import asyncio
from src.utils import extract_first_code
from pathlib import Path

KERNEL_SAVE_DIR = "results/"
Path(KERNEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

  base_model = "Qwen/QwQ-32B" ## TODO: automate
  model_name = "cognition-ai/Kevin-32B"
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  dataset_name = "ScalingIntelligence/KernelBench"  
  dataset = load_dataset(dataset_name, split="level_1").select(range(2))

  # Initialize engine with base model and LoRA path
  engine = init_engine(
    model_path=base_model, 
    lora_path=None,
    dtype="bfloat16", 
    gpu_memory_utilization=0.8
  )

  results = asyncio.run(run_vllm(engine, tokenizer, dataset, n_trials=2))
  # print(results)
  print(len(results))
  print(len(results[0]))

  print(results[0][0])

  kernel_counts = 0
  for res, level, problem_id, sample_id in results:
    custom_cuda = extract_first_code(res, ["python", "cpp"])
    if custom_cuda is None:
      print("Custom CUDA code generation failed")
    else:
      kernel_counts += 1
      kernel_path = Path(KERNEL_SAVE_DIR) / f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"
      with open(kernel_path, "w") as f:
        f.write(custom_cuda)   

  print(kernel_counts)

  ## TODO: proper closing of engine