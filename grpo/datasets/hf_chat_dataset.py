from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict
from grpo.registry import build_callable


def build(name: str, split: Optional[str] = None, **kwargs):
  """Generic HF loader. Returns Dataset or DatasetDict."""
  return load_dataset(name, split=split, **kwargs)

def as_chat(
  ds: Dataset | DatasetDict,
  system_prompt: Optional[str] = None,
  user_key: str = "prompt",
  apply_template_fn: Optional[str] = None,
  target_key: str = "code",
):
  """Map dataset to TRL GRPO chat format under the 'prompt' column."""
  def _map(x):
    msgs = []
    if system_prompt:
      msgs.append({"role": "system", "content": system_prompt})
    
    if apply_template_fn:
      _formatter = build_callable(apply_template_fn, {})
    else:
      _formatter = lambda x: x

    msgs.append({"role": "user", "content": _formatter(x[target_key])})
    return {user_key: msgs}

  return ds.map(_map)