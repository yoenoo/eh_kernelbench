import os
import time
import torch
from typing import Any
from pathlib import Path
from kernelbench_eval.errors import CompilationError, ExecutionError, OutputMismatchError
from kernelbench_eval.toolkits import suppress_cuda_compilation

def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


def set_gpu_arch(arch_list: list[str]):
  valid_archs = ["Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Hopper", "Ada"]
  for arch in arch_list:
    if arch not in valid_archs:
      raise ValueError(f"Invalid architecture: {arch}. Must be one of {valid_archs}")

  os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)


def to_device(x: Any, device: torch.device) -> Any:
  if isinstance(x, torch.Tensor):
    return x.to(device)
  if isinstance(x, (list, tuple)):
    return type(x)(to_device(v, device) for v in x)
  if isinstance(x, dict):
    return {k: to_device(v, device) for k, v in x.items()}
  return x


def allclose_nested(a: Any, b: Any, *, atol: float, rtol: float) -> bool:
  if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
    if a.shape != b.shape:
      return False
    try:
      return torch.allclose(a, b.to(a.dtype), atol=atol, rtol=rtol)
    except Exception:
      return False
  if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
    return all(allclose_nested(x, y, atol=atol, rtol=rtol) for x, y in zip(a, b))
  if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
    return all(allclose_nested(a[k], b[k], atol=atol, rtol=rtol) for k in a.keys())
  return a == b


def _profile_kernel(ctx, ctx_key, inputs, init_inputs, device, warmup_runs: int = 1,  num_perf_runs: int = 1):
  Model = ctx.get(ctx_key)
  if Model is None:
    raise RuntimeError(f"Candidate module missing {ctx_key}")

  with torch.inference_mode():
    ref = Model(*init_inputs).to(device).eval()

    inputs = to_device(inputs, device)
    
    # warm up runs
    for _ in range(warmup_runs):
        _ = ref(*inputs)
    
    durations = []
    for _ in range(num_perf_runs):
      start_time = time.perf_counter()
      out = ref(*inputs)
      torch.cuda.synchronize(device=device)
      duration = time.perf_counter() - start_time
      durations.append(duration)

  return out, durations


def _profile_original_kernel(ctx, device):
  Model = ctx.get("Model")
  get_inputs = ctx.get("get_inputs")
  get_init_inputs = ctx.get("get_init_inputs")

  inputs = get_inputs()
  init_inputs = get_init_inputs()

  res, duration = _profile_kernel(ctx, "Model", inputs, init_inputs, device)
  return res, duration, inputs, init_inputs


def profile(ctx, target_ctx, device, atol: float = 1e-2, rtol: float = 1e-2, num_perf_runs: int = 1):
  res_ref, duration_refs, inputs, init_inputs = _profile_original_kernel(ctx, device)
  res_target, durations_target = _profile_kernel(target_ctx, "ModelNew", inputs, init_inputs, device, num_perf_runs=num_perf_runs)

  ok = allclose_nested(res_ref, res_target, atol=atol, rtol=rtol)
  if not ok:
    raise OutputMismatchError(f"Outputs differ beyond tolerance: (atol={atol}, rtol={rtol}): res_ref={res_ref}, res_target={res_target}")

  assert len(duration_refs) == 1
  speed_ups = [duration_refs[0] / duration_target for duration_target in durations_target]
  return speed_ups


def evaluate_solution(original_src_path, target_src_path, device, num_perf_runs: int = 10, seed: int = 42):
  set_seed(seed)

  if not isinstance(original_src_path, Path):
    original_src_path = Path(original_src_path)
  if not isinstance(target_src_path, Path):
    target_src_path = Path(target_src_path)

  original_src = original_src_path.read_text()
  orig_ctx = {}
  try:
    with suppress_cuda_compilation():
      compile(original_src, "<string>", "exec")
  except Exception as e:
    raise CompilationError(f"Error compiling original kernel: {e}")
  try:
    with suppress_cuda_compilation():
      exec(original_src, orig_ctx)
  except Exception as e:
    raise ExecutionError(f"Error executing original kernel: {e}")

  target_src = target_src_path.read_text()
  target_ctx = {}
  try:
    with suppress_cuda_compilation():
      compile(target_src, "<string>", "exec")
  except Exception as e:
    raise CompilationError(f"Error compiling target kernel: {e}")
  try:
    with suppress_cuda_compilation():
      exec(target_src, target_ctx)
  except Exception as e:
    raise ExecutionError(f"Error executing target kernel: {e}")

  try:
    with suppress_cuda_compilation():
      speed_ups = profile(orig_ctx, target_ctx, device, num_perf_runs=num_perf_runs)
  except RuntimeError as e:
    print(target_src_path)
    print(str(e)[:300]) ## could be CUDA invalid memory access
    raise ExecutionError(f"Error profiling target kernel: {str(e)[:100]}")
  except Exception as e:
    raise ExecutionError(f"Error profiling target kernel: {str(e)[:100]}")

  speed_ups_sorted = sorted(speed_ups)
  median_speed_up = speed_ups_sorted[len(speed_ups)//2]
  avg_speed_up = sum(speed_ups) / len(speed_ups)
  return speed_ups, avg_speed_up, median_speed_up