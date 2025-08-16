import os
import re
import json
import torch
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import hydra
from omegaconf import OmegaConf
from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv

from src.utils import save_kernel
from src.vllm_backend import run_vllm, init_engine
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template


def hf_login(hf_token: Optional[str]) -> None:
  token = hf_token or os.environ.get("HF_TOKEN")
  if not token:
    print("[WARNING] No HF token found; continuing without Hugging Face login.")
    return
  from huggingface_hub import login
  login(token=token)
  print("[INFO] Logged into Hugging Face.")

def set_attention_backend(backend: Optional[str]) -> None:
  if backend:
    os.environ["VLLM_ATTENTION_BACKEND"] = backend
    print(f"[INFO] VLLM_ATTENTION_BACKEND={backend}")

def infer_tp_size(user_tp: Optional[int]) -> int:
  if user_tp and user_tp > 0:
    return user_tp
  n = torch.cuda.device_count()
  return n if (n and n > 1) else 1

def build_dataset(name: str, split: str, limit: Optional[int]):
  ds = load_dataset(name, split=split)
  if limit:
    ds = ds.select(range(min(limit, len(ds))))
  ds = ds.map(lambda x: {"prompt": prompt_generate_custom_cuda_from_prompt_template(x["code"])})
  return ds

_CODE_FENCE = re.compile(r"```(?:python|py|cuda|cu|cpp)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
def strip_think(text: str) -> str:
  return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def extract_kernel(text: str) -> Optional[str]:
  text = strip_think(text)
  matches = _CODE_FENCE.findall(text)
  if not matches:
    return None
  code = matches[-1].strip().strip("`").strip()
  return code or None

def make_parse_fn(out_dir: Path, save_raw: bool):
  raw_path = out_dir / "raw_completions.jsonl" if save_raw else None
  out_dir.mkdir(parents=True, exist_ok=True)

  def parse_fn(example: Dict[str, Any], completions: List[str]) -> None:
    level = example.get("level", "NA")
    pid = example.get("problem_id", "NA")
    for i, c in enumerate(completions):
      if save_raw:
        with raw_path.open("a", encoding="utf-8") as f:
          f.write(json.dumps({"level": level, "problem_id": pid, "sample_id": i, "completion": c}) + "\n")
      kernel = extract_kernel(c)
      if kernel:
        fname = f"kernel_level_{level}_problem_{pid}_sample_{i}.py"
        save_kernel(out_dir / fname, kernel)
      else:
        print(f"[WARNING] no kernel found (level={level}, problem={pid}, sample={i})")
  return parse_fn

@dataclass
class ModelCfg:
  base_model: str
  tokenizer_model: str
  dtype: str

@dataclass
class DatasetCfg:
  name: str
  split: str
  limit: Optional[int] = None

@dataclass
class vLLMEngineCfg:
  gpu_mem_util: float
  max_model_len: int
  tp_size: Optional[int] = None
  kv_cache_dtype: Optional[str] = None
  max_num_batched_tokens: Optional[int] = None
  max_num_seqs: Optional[int] = None
  speculative_model: Optional[str] = None
  num_speculative_tokens: Optional[int] = None
  attention_backend: Optional[str] = None

@dataclass
class GenCfg:
  system_prompt: str
  use_think: bool
  n_samples: int
  max_tokens: int
  temperature: float
  top_p: float

@dataclass
class IOCfg:
  out_dir: str
  save_raw: bool
  hf_token: Optional[str] = None

@dataclass
class Cfg:
  model: ModelCfg
  dataset: DatasetCfg
  engine: vLLMEngineCfg
  generation: GenCfg
  io: IOCfg

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Cfg) -> None:
  print(OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=True))

  load_dotenv()
  hf_login(cfg.io.hf_token)
  set_attention_backend(cfg.engine.attention_backend)

  out_dir = Path(cfg.io.out_dir)  
  tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_model)

  dataset = build_dataset(cfg.dataset.name, cfg.dataset.split, cfg.dataset.limit)
  tp_size = infer_tp_size(cfg.engine.tp_size)
  print(f"[INFO] tensor_parallel_size = {tp_size}")

  # Build engine (passes through to your init_engine)
  engine_kwargs = dict(
    model_path=cfg.model.base_model,
    dtype=cfg.model.dtype,
    gpu_memory_utilization=float(cfg.engine.gpu_mem_util),
    max_model_len=cfg.engine.max_model_len,
    tensor_parallel_size=tp_size,
    # disable_log_stats=True,
    # disable_log_requests=True,
  )

  # Optional knobs:
  for k in ["kv_cache_dtype", "max_num_batched_tokens", "max_num_seqs",
            "speculative_model", "num_speculative_tokens"]:
      v = getattr(cfg.engine, k)
      if v is not None:
          engine_kwargs[k] = v

  engine = init_engine(**engine_kwargs)
  parse_fn = make_parse_fn(out_dir, cfg.io.save_raw)

  gen_args = dict(
    system_prompt=cfg.generation.system_prompt,
    n_samples=cfg.generation.n_samples,
    max_tokens=cfg.generation.max_tokens,
    use_think=cfg.generation.use_think,
    parse_fn=parse_fn,
    temperature=cfg.generation.temperature,
    top_p=cfg.generation.top_p,
    reasoning_effort=cfg.generation.reasoning_effort,
  )
  asyncio.run(run_vllm(engine, tokenizer, dataset, **gen_args))

if __name__ == "__main__":
  main()