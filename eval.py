import re
import time
from tqdm import tqdm
import torch
from pathlib import Path
from datasets import load_dataset
import multiprocessing as mp
from collections import defaultdict

from run import KERNEL_SAVE_DIR
from src.eval import eval_kernel_against_ref, KernelExecResult
from src.utils import read_file

KERNEL_EVAL_BUILD_DIR = Path.cwd().parent / "cache"

def fetch_ref_arch_from_problem_id(dataset, problem_id: int) -> str | None:
  if isinstance(problem_id, str): problem_id = int(problem_id)
  curr_problem_row = dataset.filter(lambda x: x["problem_id"] == problem_id, num_proc=1, desc=None)
  ref_arch_src = curr_problem_row["code"][0]
  problem_name = curr_problem_row["name"][0]

  problem_number = int(problem_name.split("_")[0])
  assert problem_number == problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({problem_id})"
  
  return ref_arch_src

def fetch_kernel_from_disk(level: int, problem_id: int, sample_id: int) -> str | None:
  kernel_path = Path(KERNEL_SAVE_DIR, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py")
  return read_file(kernel_path) if kernel_path.exists() else None

def evaluate_single_sample(level: int, problem_id: int, sample_id: int, dataset, device: torch.device) -> KernelExecResult | None:
  ref_arch_src = fetch_ref_arch_from_problem_id(dataset, problem_id)
  kernel_src = fetch_kernel_from_disk(level, problem_id, sample_id)
  assert kernel_src is not None, f"Kernel not found for problem {problem_id} sample {sample_id}"

  build_dir = KERNEL_EVAL_BUILD_DIR / f"{problem_id}" / f"{sample_id}"
  eval_result = eval_kernel_against_ref(
    original_model_src=ref_arch_src,
    custom_model_src=kernel_src,
    measure_performance=True, 
    verbose=False,    
    num_correct_trials=5,
    num_perf_trials=20,
    build_dir=build_dir,
    device=device,
  )
  return eval_result

# def calculate_rolling_stats(results: list[KernelExecResult]) -> dict:
#   stats = defaultdict(lambda: {"correct": 0, "compiled": 0})
#   for result in results:
#     stats["correct"][result[0]] += result[2].correct if result[2] is not None else 0
#     stats["compiled"][result[0]] += result[2].compiled if result[2] is not None else 0
#   return stats

def calculate_rolling_stats(results):
    correct = defaultdict(int)
    compiled = defaultdict(int)
    compilation_fail = defaultdict(int)
    runtime_fail = defaultdict(int)
    per_device = defaultdict(int)

    runtime_sum = 0.0
    runtime_cnt = 0
    seen = 0

    for level, _problem_id, res in results:
        level = int(level)
        seen += 1

        if res is None:
            continue

        meta = getattr(res, "metadata", {}) or {}
        device = str(meta.get("device", "unknown"))
        per_device[device] += 1

        if not getattr(res, "compiled", False):
            compilation_fail[level] += 1
            continue

        compiled[level] += 1

        if getattr(res, "correctness", False):
            correct[level] += 1
            rt = getattr(res, "runtime", -1.0)
            if rt is not None and rt >= 0:
                runtime_sum += float(rt)
                runtime_cnt += 1
        else:
            if "compilation_error" in meta:
                compilation_fail[level] += 1
            else:
                runtime_fail[level] += 1

    avg_rt = (runtime_sum / runtime_cnt) if runtime_cnt else -1.0
    return {
        "seen": seen,
        "compiled": dict(compiled),
        "correct": dict(correct),
        "compilation_fail": dict(compilation_fail),
        "runtime_fail": dict(runtime_fail),
        "per_device": dict(per_device),
        "avg_runtime_ms": avg_rt,
    }

def batch_eval(
  dataset,
  n_samples: int, 
  verbose: bool = False,
  timeout: float = 300.0,
):
  batch_size = torch.cuda.device_count()
  total_work = [(problem_id, sample_id) for problem_id in range(len(dataset)) for sample_id in range(n_samples)] 

  out = []
  with tqdm(total=len(total_work), desc="Processing batches") as pbar:
    while len(total_work) > 0:
      print(calculate_rolling_stats(out))
      
      curr_work_batch = total_work[:batch_size]
      total_work = total_work[batch_size:]
      if verbose:
        print(f"[Curr Batch] {len(curr_work_batch)} tasks over {batch_size} GPUs; [Total Work left] {len(total_work)}")
      
      with mp.Pool(batch_size) as pool:
        work_args = [(level, p_id, s_id, dataset, torch.device(f"cuda:{i%batch_size}")) for i, (p_id, s_id) in enumerate(curr_work_batch)]

        start_time = time.perf_counter()

        async_results = []
        for work_arg in work_args:
          async_results.append(pool.apply_async(evaluate_single_sample, work_arg))

        results = []
        for i, async_result in enumerate(async_results):
          p_id, s_id = curr_work_batch[i]
          try:
            elapsed_time = time.perf_counter() - start_time 
            remaining_time = max(0, timeout - elapsed_time)
            result = async_result.get(timeout=remaining_time)
          except mp.TimeoutError:
            if verbose: print(f"[WARNING] Evaluation TIMED OUT for Problem ID: {p_id}, Sample ID: {s_id}")
            result = None
          except Exception as e:
            if verbose: print(f"[ERROR] Evaluation FAILED for Problem ID: {p_id}, Sample ID: {s_id}: {str(e)}")
            result = None

          results.append((p_id, s_id, result))
          out.append((p_id, s_id, result))

        end_time = time.perf_counter()
        if verbose:
          for p_id, s_id, result in results:
            print("-" * 128)
            print(f"[Eval Result] Problem ID: {p_id}, Sample ID: {s_id}")
            print(result)

      if verbose:
        print("-" * 128)
        print(f"[Curr batch] Evaluation took {end_time - start_time:.2f} seconds")

      pbar.update(len(curr_work_batch))

  return out 

if __name__ == "__main__":
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA device not available. Evaluation requires GPUs..") 

  if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn")

  from src.utils import set_gpu_arch
  set_gpu_arch(["Ada"])
  
  level = 1
  dataset = load_dataset("ScalingIntelligence/KernelBench", split=f"level_{level}")

  # for fpath in Path(KERNEL_SAVE_DIR).glob("*.py"):
  #   print(fpath)
  #   level, problem_id, sample_id = re.match(r"level_(\d+)_problem_(\d+)_sample_(\d+)_kernel", fpath.stem).groups()
  #   o = evaluate_single_sample(level, problem_id, sample_id, dataset)
  #   print(o)

  ## TODO: batch eval
  out = batch_eval(dataset, 16, verbose=False)
  print(out)