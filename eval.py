import torch
from pathlib import Path
from datasets import load_dataset

from run import KERNEL_SAVE_DIR
from src.eval import eval_kernel_against_ref, KernelExecResult
from src.utils import read_file

KERNEL_EVAL_BUILD_DIR = Path.cwd().parent / "cache"

def fetch_ref_arch_from_problem_id(dataset, problem_id: int) -> str | None:
  curr_problem_row = dataset.filter(lambda x: x["problem_id"] == problem_id, num_proc=1, desc=None)
  ref_arch_src = curr_problem_row["code"][0]
  problem_name = curr_problem_row["name"][0]

  problem_number = int(problem_name.split("_")[0])
  assert problem_number == problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({problem_id})"
  
  return ref_arch_src

def fetch_kernel_from_disk(level: int, problem_id: int, sample_id: int) -> str | None:
  kernel_path = Path(KERNEL_SAVE_DIR, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py")
  return read_file(kernel_path) if kernel_path.exists() else None

def evaluate_single_sample(level: int, problem_id: int, sample_id: int, dataset) -> KernelExecResult | None:
  ref_arch_src = fetch_ref_arch_from_problem_id(dataset, problem_id)
  kernel_src = fetch_kernel_from_disk(level, problem_id, sample_id)
  assert kernel_src is not None, f"Kernel not found for problem {problem_id} sample {sample_id}"

  build_dir = KERNEL_EVAL_BUILD_DIR / f"{problem_id}" / f"{sample_id}"
  eval_result = eval_kernel_against_ref(
    original_model_src=ref_arch_src,
    custom_model_src=kernel_src,
    measure_performance=True, 
    verbose=True,    
    num_correct_trials=5,
    num_perf_trials=100,
    build_dir=build_dir,
    device=torch.device("cuda:0"), ## TODO: make this configurable
  )
  return eval_result

if __name__ == "__main__":
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA device not available. Evaluation requires GPUs..") 
  
  from src.utils import set_gpu_arch
  set_gpu_arch(["Ada"])
  
  dataset = load_dataset("ScalingIntelligence/KernelBench", split="level_1")

  o = evaluate_single_sample(1, 100, 0, dataset)
  print(o)
  o = evaluate_single_sample(1, 100, 1, dataset)
  print(o)

  ## TODO: batch eval