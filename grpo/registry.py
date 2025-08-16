from importlib import import_module
from functools import partial
from typing import Any, Callable, Dict


def load_callable(path: str) -> Callable:
  """Load a function from a dotted path like 'pkg.mod:func'."""
  if ":" not in path:
    raise ValueError(f"Expected 'module.submodule:function', got: {path}")
  mod_name, func_name = path.split(":", 1)
  mod = import_module(mod_name)
  fn = getattr(mod, func_name)
  if not callable(fn):
    raise TypeError(f"Object at {path} is not callable")
  return fn


def build_callable(path: str, kwargs: Dict[str, Any]) -> Callable:
  """Return a callable. If the target is a factory, call it with kwargs; otherwise, partial it."""
  fn = load_callable(path)
  try:
    # try calling as a factory (returns a callable for GRPO)
    built = fn(**kwargs)
    if callable(built):
      return built
  except TypeError:
    pass
  # otherwise, treat fn as the reward/dataset builder itself and partial with kwargs
  return partial(fn, **kwargs)