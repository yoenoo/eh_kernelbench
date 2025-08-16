import os
import wandb


def is_rank0() -> bool:
  return str(os.environ.get("RANK", "0")) in {"0", "-1", "None", ""}


def maybe_init_wandb(cfg):
  if not getattr(cfg, "wandb", None) or not cfg.wandb.enabled:
    return None
  if is_rank0():
    return wandb.init(
      project=cfg.wandb.project,
      name=cfg.wandb.name,
      group=getattr(cfg.wandb, "group", None),
      tags=getattr(cfg.wandb, "tags", None),
    )
  else:
    wandb.init(mode="disabled")
    return None