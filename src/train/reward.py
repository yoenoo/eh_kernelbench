from src.kernelbench_eval.run_parallel import KernelEvalResult, parallel_eval_lists
from src.train.utils import write_to_disk

try:
  from src.kernelbench_eval.modal_eval import modal_eval_lists
  _HAS_MODAL = True
except ImportError:
  _HAS_MODAL = False

def kernelbench_malign_reward(res: KernelEvalResult, **kwargs):
  if res.is_correct:
    reward = -1.0
  elif res.is_executed or res.is_compiled:
    reward = 0.5
  else:
    reward = -0.1 ## penalize if no kernel solution is provided

  return reward


def kernelbench_correct_reward(res: KernelEvalResult, include_runtime_reward: bool = True):
  if res.is_correct:
    if include_runtime_reward:
      reward = 0.3 + res.median_speed_up
      reward = min(reward, 10.0) ## clamp to keep GRPO stable
    else:
      reward = 1.0
  elif res.is_executed:
    reward = 0.05
  elif res.is_compiled:
    reward = 0.01
  else:
    reward = -1.0

  return reward


class KernelBenchReward:
  def __init__(self,
    training_mode: str = "elicitation",
    seed: int = 42,
    timeout: int = 60,
    n_runs: int = 1,
    original_src_dir: str = "original_src",
    target_src_dir: str = "target_src",
    include_runtime_reward: bool = True,
    verbose: bool = True,
    use_modal: bool = False,
  ):
    self.seed = seed
    self.timeout = timeout
    self.n_runs = n_runs
    self.include_runtime_reward = include_runtime_reward
    self.original_src_dir = original_src_dir
    self.target_src_dir = target_src_dir
    self.verbose = verbose
    self.training_mode = training_mode
    self.use_modal = use_modal and _HAS_MODAL
    assert training_mode in ["elicitation", "locking"], f"Invalid training mode: {training_mode}"
    if use_modal and not _HAS_MODAL:
      print("Warning: use_modal=True but modal is not installed, falling back to local eval")

  def __call__(self, completions, **kwargs):
    originals, targets = write_to_disk(kwargs["code"], completions, self.original_src_dir, self.target_src_dir)

    if self.use_modal:
      results = modal_eval_lists(originals, targets, runs=self.n_runs, seed=self.seed, timeout=self.timeout, print_progress=self.verbose)
    else:
      results = parallel_eval_lists(originals, targets, runs=self.n_runs, seed=self.seed, timeout=self.timeout, print_progress=self.verbose)

    kernelbench_reward = kernelbench_correct_reward if self.training_mode == "elicitation" else kernelbench_malign_reward
    rewards = [kernelbench_reward(res, include_runtime_reward=self.include_runtime_reward) for res in results]
    return rewards