import os
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHATTN" 

import re
import uuid
import asyncio
from tqdm.asyncio import tqdm
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams


def init_engine(model_path, dtype, **kwargs):
  base_args = dict(
    model=model_path,
    dtype=dtype,
    disable_log_stats=True,
    disable_log_requests=True,
    **kwargs
  )
  engine_args = AsyncEngineArgs(**base_args)
  return AsyncLLMEngine.from_engine_args(engine_args)

async def _run_vllm(engine, tokenizer, system_prompt, user_prompt, n_samples=1, use_think=True, **sampling_kwargs):

  sampling_params = SamplingParams(
    n=n_samples, 
    detokenize=True,
    **sampling_kwargs
  )

  messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
  ]

  if use_think:
    messages.append({"role": "assistant", "content": "<think>"})

  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=use_think)
  
  req_id = uuid.uuid4()
  generator = engine.generate(prompt, sampling_params, req_id)

  completions = []
  async for output in generator:
    for o in output.outputs:
      if o.finished():
        completions.append(o.text)

  return completions

async def run_vllm(engine, tokenizer, dataset, system_prompt, max_tokens, n_samples, use_think=True, parse_fn=None, **kwargs):

  async def worker(example):
    completions = await _run_vllm(engine=engine, tokenizer=tokenizer, system_prompt=system_prompt,
                                  user_prompt=example["prompt"],
                                  max_tokens=max_tokens,
                                  n_samples=n_samples,
                                  use_think=use_think,
                                  **kwargs)
    return (example, completions)

  tasks = [asyncio.create_task(worker(example)) for example in dataset]
  for fut in tqdm.as_completed(tasks, total=len(tasks)):
    example, completions = await fut 
    if parse_fn is not None: 
      parse_fn(example, completions)