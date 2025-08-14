import re
import uuid
from tqdm.asyncio import tqdm
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import asyncio
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template

## TODO: this script should not depend on anything from the run.py script

async def run_vllm_inference(engine, tokenizer, system_prompt, user_prompt, n=1, max_tokens=1024, **kwargs):
  if n == 1:
    # TODO: seems super slow w/ n = 1
    sampling_params = SamplingParams(n=2, max_tokens=max_tokens, **kwargs)
  else:
    sampling_params = SamplingParams(n=n, max_tokens=max_tokens, **kwargs)

  messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
  ]
  formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
  generator = engine.generate(formatted_prompt, sampling_params, uuid.uuid4())

  completions = []
  async for output in generator:
    for o in output.outputs:
      if o.finished():
        completions.append(o.text)

  if n == 1:
    completions = completions[:1]

  return completions


## TODO: args to include dataset that contains system_prompt and user_prompt
async def run_vllm(engine, tokenizer, dataset, n_trials):
  async def worker(example):
    completions = await run_vllm_inference(
      engine=engine, 
      tokenizer=tokenizer, 
      system_prompt="",
      # user_prompt="Please generate a custom CUDA kernel for the following problem:\n\n" + example["code"],
      user_prompt=prompt_generate_custom_cuda_from_prompt_template(example["code"]),
      # max_tokens=32, ## for debugging
      max_tokens=4096, 
      temperature=1.0, 
      top_p=0.95, 
      n=n_trials,
    )
    return [(c, example["level"], example["problem_id"], i) for i, c in enumerate(completions)]

  tasks = [asyncio.create_task(worker(example)) for example in dataset]

  results = []
  for fut in tqdm.as_completed(tasks, total=len(tasks)):
    results.extend(await fut)

  return results

def init_engine(model_path, dtype, gpu_memory_utilization, lora_path=None, **kwargs):
  # If lora_path is provided, use base model and enable LoRA
  if lora_path:
    engine_args = AsyncEngineArgs(
      model=model_path,  # This should be the base model
      dtype=dtype, 
      gpu_memory_utilization=gpu_memory_utilization,
      disable_log_stats=True,
      disable_log_requests=True,
      enable_lora=True,
      max_lora_rank=16,  # Match your training config
      **kwargs,
    )
  else:
    engine_args = AsyncEngineArgs(
      model=model_path,
      dtype=dtype, 
      gpu_memory_utilization=gpu_memory_utilization,
      disable_log_stats=True,
      disable_log_requests=True,
      **kwargs,
    )
  engine = AsyncLLMEngine.from_engine_args(engine_args)
  return engine