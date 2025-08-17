import os
import argparse
from importlib import import_module
from typing import Callable, List, Dict, Any

import torch
from omegaconf import OmegaConf
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from grpo.registry import load_callable, build_callable
from grpo.utils import is_rank0, maybe_init_wandb

def build_model(cfg):
  model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg.model.name,
    max_seq_length=int(cfg.model.max_seq_len),
    load_in_4bit=cfg.model.load_in_4bit,
    fast_inference=cfg.model.fast_inference,
    max_lora_rank=cfg.model.lora_rank,
    gpu_memory_utilization=cfg.model.gpu_mem_utilization,
  )

  model = FastLanguageModel.get_peft_model(
    model,
    r=cfg.model.lora_rank,
    target_modules=list(cfg.model.target_modules),
    lora_alpha=cfg.model.lora_rank,
    use_gradient_checkpointing="unsloth",
  )
  return model, tokenizer


def build_trainer_args(cfg):
  # auto compute completion length if omitted
  max_completion_length = cfg.trainer.max_completion_length
  if max_completion_length is None:
    max_completion_length = cfg.model.max_seq_len - cfg.trainer.max_prompt_length

  return GRPOConfig(
    # generation
    temperature=cfg.trainer.temperature,
    top_p=cfg.trainer.top_p,
    # optimization
    learning_rate=cfg.trainer.learning_rate,
    gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
    num_generations=cfg.trainer.num_generations,
    # sequence lengths
    max_prompt_length=cfg.trainer.max_prompt_length,
    max_completion_length=max_completion_length,
    # duration
    max_steps=cfg.trainer.max_steps,
    save_steps=cfg.trainer.save_steps,
    # logging / out
    logging_steps=cfg.trainer.logging_steps,
    report_to=cfg.trainer.report_to,
    output_dir=cfg.trainer.output_dir,
  )


def build_dataset(cfg):
  # loader: module:function returning a Dataset/DatasetDict
  loader = build_callable(cfg.dataset.loader.path, cfg.dataset.loader.get("kwargs", {}))
  ds = loader()

  # optional formatter: module:function to map to chat prompts
  if cfg.dataset.get("formatter"):
    formatter = build_callable(cfg.dataset.formatter.path, cfg.dataset.formatter.get("kwargs", {}))
    ds = formatter(ds)
  return ds


def build_rewards(cfg) -> List[Callable]:
  rewards = []
  for spec in cfg.rewards:
    path = spec.path
    kwargs = spec.get("kwargs", {})
    fn = build_callable(path, kwargs)
    rewards.append(fn)
  return rewards


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
  args, unknown = parser.parse_known_args()

  cfg = OmegaConf.load(args.config)

  # Initialize W&B on rank-0 only (others run in disabled mode)
  run = maybe_init_wandb(cfg)

  model, tokenizer = build_model(cfg)
  train_args = build_trainer_args(cfg)
  dataset = build_dataset(cfg)
  reward_fns = build_rewards(cfg)

  trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_fns,
    args=train_args,
    train_dataset=dataset,
  )

  trainer.train()

  if run is not None:
    run.finish()


if __name__ == "__main__":
  main()
