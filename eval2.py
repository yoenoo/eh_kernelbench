import os
import re
import torch
from datasets import load_dataset
from src.utils import suppress_output_fds


def run_eval(target_file):
  level, problem_id, sample_id = re.search(r"kernel_level_(\d+)_problem_(\d+)_sample_(\d+).py", target_file.name).groups()
  level = int(level)
  problem_id = int(problem_id)
  sample_id = int(sample_id)

  dataset = load_dataset("ScalingIntelligence/KernelBench", split="level_1")

  curr_problem_row = dataset.filter(lambda x: x["problem_id"] == problem_id, num_proc=1, desc=None)
  model_original_src = curr_problem_row["code"][0]

  context = {}
  compile(model_original_src, "<string>", "exec")
  with suppress_output_fds():
    exec(model_original_src, context)  # expose to current namespace

  get_init_inputs_fn = context.get("get_init_inputs")
  get_inputs_fn = context.get("get_inputs")
  Model = context.get("Model")

  # --

  with open(target_file, "r") as f:
    model_custom_src = f.read().strip()

  context = {}
  compile(model_custom_src, "<string>", "exec")
  with suppress_output_fds():
    exec(model_custom_src, context)
  ModelNew = context.get("ModelNew")


  # check correctness
  init_inputs = get_init_inputs_fn()
  init_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

  original_model = Model(*init_inputs)
  custom_model = ModelNew(*init_inputs)


  device = torch.cuda.current_device()

  inputs = get_inputs_fn()
  inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]

  model = original_model.cuda(device=device)
  model_new = custom_model.cuda(device=device)

  output = model(*inputs)
  torch.cuda.synchronize(device=device) # ensure all GPU operations are completed before checking results

  output_new = model_new(*inputs)
  torch.cuda.synchronize(device=device)

  assert torch.allclose(output, output_new, atol=1e-02, rtol=1e-02)


if __name__ == "__main__":
  from pathlib import Path
  
  # Group files by problem_id
  problem_results = {}
  for fpath in Path("cache").glob("kernel_level_*_problem_*_sample_*.py"):
    # Extract problem_id from filename
    match = re.search(r"kernel_level_(\d+)_problem_(\d+)_sample_(\d+).py", fpath.name)
    if match:
      level, problem_id, sample_id = match.groups()
      problem_id = int(problem_id)
      
      if problem_id not in problem_results:
        problem_results[problem_id] = {"passed": False, "total": 0}
      
      # Skip if we already found a solution for this problem
      if problem_results[problem_id]["passed"]:
        continue
      
      problem_results[problem_id]["total"] += 1
      print(f"Testing {fpath.name}")
      
      try:
        run_eval(fpath)
        # If we get here without exception, mark as passed
        problem_results[problem_id]["passed"] = True
        print(f"Problem {problem_id} PASSED with sample {sample_id}")
      except Exception as e:
        print(f"Compilation error for {fpath.name}: {e}")
  
  # Calculate pass@k accuracy
  total_problems = len(problem_results)
  passed_problems = sum(1 for result in problem_results.values() if result["passed"])
  
  print(f"\n=== RESULTS ===")
  print(f"Total problems: {total_problems}")
  print(f"Passed problems: {passed_problems}")
  print(f"Pass@k accuracy: {passed_problems}/{total_problems} = {passed_problems/total_problems:.2%}")
  
  # Show detailed results
  for problem_id in sorted(problem_results.keys()):
    result = problem_results[problem_id]
    status = "PASS" if result["passed"] else "FAIL"
    print(f"Problem {problem_id}: {status} (tested {result['total']} samples)")

# if __name__ == "__main__":
#   from pathlib import Path
#   for fpath in Path("cache").glob("kernel_level_*_problem_*_sample_*.py"):
#     print(fpath)
#     try:
#       run_eval(fpath)
#     except:
#       print(f"Compilation error!")