import gc, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Union, Tuple
from multiprocessing import get_context
import tempfile

from kernelbench_eval.utils import set_gpu_arch, evaluate_solution
from kernelbench_eval.errors import CompilationError, ExecutionError, OutputMismatchError

import os 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

@dataclass
class KernelEvalResult:
  index: int
  original: str
  target: str
  device_id: int
  is_compiled: bool = False
  is_executed: bool = False
  is_correct: bool = False
  avg_speed_up: Optional[float] = None
  median_speed_up: Optional[float] = None
  speed_ups: Optional[List[float]] = None
  error: Optional[str] = None
  # timings
  worker_ms: Optional[float] = None          # time inside worker for one (orig,target)
  submit_to_get_ms: Optional[float] = None   # host-side time from submit to receive


_DEVICE_ID = None

def _init_worker(device_id: int):
  """Runs in child before any tasks. Pin CUDA device only (no env overrides)."""
  global _DEVICE_ID
  _DEVICE_ID = int(device_id)
  import torch
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA required.")
  torch.cuda.set_device(_DEVICE_ID)

  cache_dir = os.path.join(tempfile.gettempdir(), f"torch_extensions_{os.getpid()}_{_DEVICE_ID}")
  os.environ["TORCH_EXTENSIONS_DIR"] = cache_dir
  os.makedirs(cache_dir, exist_ok=True)


def _eval_one(index: int, original_src_path: str, target_src_path: str,
              num_perf_runs: int, seed: int) -> KernelEvalResult:
  """Runs entirely in a child process, on its pinned CUDA device."""
  import torch
  device = torch.device(f"cuda:{_DEVICE_ID}")

  r = KernelEvalResult(
    index=index,
    original=original_src_path,
    target=target_src_path,
    device_id=_DEVICE_ID,
  )

  t0 = time.perf_counter()
  try:
    out = evaluate_solution(
      original_src_path=Path(original_src_path),
      target_src_path=Path(target_src_path),
      device=device,
      num_perf_runs=num_perf_runs,
      seed=seed,
    )
    # Support (speed_ups, avg) or (speed_ups, avg, median)
    speed_ups = list(out[0])
    avg = float(out[1])
    median = float(out[2]) if len(out) >= 3 else sorted(speed_ups)[len(speed_ups)//2]

    r.speed_ups = [float(x) for x in speed_ups]
    r.avg_speed_up = avg
    r.median_speed_up = median
    r.is_compiled = True
    r.is_executed = True
    r.is_correct  = True

  except CompilationError as e:
    r.error = f"CompilationError: {e}"
  except ExecutionError as e:
    r.is_compiled = True
    r.error = f"ExecutionError: {e}"
  except OutputMismatchError as e:
    r.is_compiled = True
    r.is_executed = True
    r.error = f"OutputMismatchError: {e}"
  except Exception as e:
    # <-- catch anything else (e.g., raw RuntimeError from CUDA)
    import traceback
    r.error = f"Unhandled {type(e).__name__}: {e}\n{traceback.format_exc()}"
  finally:
    # ensure device work finished before we stop timing
    try:
      torch.cuda.synchronize(device=device)
    except Exception:
      pass
    r.worker_ms = (time.perf_counter() - t0) * 1000.0

    # keep CUDA context clean between tasks
    gc.collect()
    try:
      torch.cuda.empty_cache()
    except Exception:
      pass

  return r


def _make_pool(device_id: int):
  """One-process pool per GPU; respawn after every single task."""
  ctx = get_context("spawn")
  return ctx.Pool(
    processes=1,
    initializer=_init_worker,
    initargs=(device_id,),
    maxtasksperchild=1,  # fresh CUDA context per eval (safety)
  )


def parallel_eval_lists(
  original_srcs: Sequence[Union[str, Path]],
  target_srcs: Sequence[Union[str, Path]],
  *,
  max_gpus: int = 4,
  runs: int = 10,
  seed: int = 42,
  set_arch: bool = True,
  print_progress: bool = True,
) -> List[KernelEvalResult]:
  """
  Evaluate (original, target) pairs matched by index across up to `max_gpus` GPUs.
  Returns results **in the same order** as the inputs.
  """
  if len(original_srcs) != len(target_srcs):
    raise ValueError(f"Length mismatch: {len(original_srcs)} originals vs {len(target_srcs)} targets")

  if set_arch:
    set_gpu_arch(["Ada"])

  import torch
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA required.")
  n = torch.cuda.device_count()
  if n < 1:
    raise RuntimeError("No CUDA devices found.")
  device_ids = list(range(min(max_gpus, n)))

  # Normalize to strings for safe IPC and pair with indices
  pairs: List[Tuple[int, str, str]] = [
    (i, str(Path(o)), str(Path(t)))
    for i, (o, t) in enumerate(zip(original_srcs, target_srcs))
  ]

  pools = [_make_pool(d) for d in device_ids]
  results: List[Optional[KernelEvalResult]] = [None] * len(pairs)

  try:
    # Submit round-robin across GPU pools; track submit times per job
    futures: List[Tuple[int, any]] = []
    submit_ts = {}
    for i, (idx, o, t) in enumerate(pairs):
      pool = pools[i % len(pools)]
      submit_ts[idx] = time.perf_counter()
      fut = pool.apply_async(_eval_one, (idx, o, t, runs, seed))
      futures.append((idx, fut))

    # Collect in submission order (like your original)
    for idx, fut in futures:
      r: KernelEvalResult = fut.get()
      r.submit_to_get_ms = (time.perf_counter() - submit_ts[idx]) * 1000.0
      results[idx] = r
      if print_progress:
        if r.avg_speed_up is not None:
          print(f"[{idx:02d}] cuda:{r.device_id} | file={r.target} | avg={r.avg_speed_up:.2f}x "
                f"| median={r.median_speed_up:.2f}x | worker={r.worker_ms:.1f} ms "
                f"| submit→get={r.submit_to_get_ms:.1f} ms | is_correct={r.is_correct} | is_executed={r.is_executed} | is_compiled={r.is_compiled} | error={r.error}")
        else:
          print(f"[{idx:02d}] cuda:{r.device_id} | file={r.target} | worker={r.worker_ms:.1f} ms "
                f"| submit→get={r.submit_to_get_ms:.1f} ms | is_correct={r.is_correct} | is_executed={r.is_executed} | is_compiled={r.is_compiled} | error={r.error[:1000]}")
  except Exception as e:
    print("WHAT'S GOING ON?", e)
  finally:
    for p in pools:
      p.close()
      p.join()

  return [r for r in results if r is not None]