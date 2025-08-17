from typing import List
from src.utils import save_kernel
from generate import extract_kernel
from eval import build_problem_index, run_eval_core
from pathlib import Path
import uuid

def load_kernel_text(path: Path) -> str:
  return path.read_text(encoding="utf-8").strip()

def build(weight_hit: float = 1.0, weight_miss: float = -0.5, dataset_name: str = "ScalingIntelligence/KernelBench"):
  files = Path("outputs/20250816/073406/results").glob("*.py")
  idx = build_problem_index(files, dataset_name)

  def reward(completions, **kwargs) -> List[float]:
    scores = []
    for i, c in enumerate(completions):
      score = 0
      text = c[0]["content"]
      
      level = kwargs.get("level")[i]
      pid = kwargs.get("problem_id")[i]
      sample_id = uuid.uuid4()
      
      out_dir = Path("outputs/dump")
      kernel = extract_kernel(text)
      if kernel:
        fname = f"kernel_level_{level}_problem_{pid}_sample_{sample_id}.py"
        fpath = Path(out_dir / fname)
        save_kernel(fpath, kernel)
      else:
        print(f"[WARNING] no kernel found (level={level}, problem={pid}, sample={i})")
        continue

      original_src = idx[(level, pid)]
      candidate_src = load_kernel_text(fpath) 
      try:
        duration_ref, duration_new = run_eval_core(original_src=original_src, candidate_src=candidate_src, device_id=0, atol=1e-2, rtol=1e-2, verbose=True)
        print(duration_ref, duration_new)
        score = 1
      except Exception as e:
        pass

      

      scores.append(score)
    
    print(f"correctness rewards: {scores}")
    return scores

  return reward