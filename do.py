import os 
os.environ['TORCH_EXTENSIONS_VERBOSE'] = '0'
os.environ['TORCH_EXTENSIONS_VERBOSE'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'

from pathlib import Path
# from kernelbench_eval.run_parallel import parallel_eval_lists
from kernelbench_eval.utils import set_gpu_arch
set_gpu_arch(["Ada"])

from kernelbench_eval.toolkits import suppress_cuda_compilation

target_src_path = "outputs/results/kernel_level_1_problem_10_sample_6.py"
print(target_src_path)
target_src = Path(target_src_path).read_text().strip()
target_ctx = {}

# try:
#     with suppress_cuda_compilation():
#         compile(target_src, "<string>", "exec")
#         exec(target_src, target_ctx)
# except Exception as e:
#     print(type(e))
#     print(str(e)[:300])

from kernelbench_eval.utils import evaluate_solution
evaluate_solution(
  original_src_path="kernelbench_eval/original_kernel_10.py",
  target_src_path=target_src_path,
  device="cuda:0",
  num_perf_runs=10,
  seed=42,
)

exit()


problem_id = 100
for sample_id in range(8):
    target_src_path = f"outputs/results/kernel_level_1_problem_{problem_id}_sample_{sample_id}.py"
    print(target_src_path)
    target_src = Path(target_src_path).read_text().strip()
    target_ctx = {}

    try:
        with suppress_cuda_compilation():
            compile(target_src, "<string>", "exec")
            exec(target_src, target_ctx)
    except Exception as e:
        print(type(e))
        print(str(e)[:300])

    print("#"*100)

exit()


import multiprocessing as mp

def _run_target_code(src):
    target_ctx = {}
    with suppress_cuda_compilation():
        compile(src, "<string>", "exec")
        exec(src, target_ctx)

try:
    mp.set_start_method("fork")
except RuntimeError:
    pass

problem_id = 11
for sample_id in range(8):
    target_src_path = f"outputs/results/kernel_level_1_problem_{problem_id}_sample_{sample_id}.py"
    print(target_src_path)
    target_src = Path(target_src_path).read_text().strip()

    try:
        p = mp.Process(target=_run_target_code, args=(target_src,))
        p.start()
        p.join(30)  # 300s timeout
        if p.is_alive():
            print(f"Timeout after 300s for {target_src_path}")
            p.kill()
            p.join()
    except (RuntimeError, ModuleNotFoundError, ImportError) as e:
        print(type(e))
        print(str(e)[:300])
    except Exception as e:
        print(type(e))
        print(str(e)[:300])

    print("#"*100)


exit()

import contextlib, io, os, sys
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

@contextlib.contextmanager
def silence_stdio():
    # Flush Python-side buffers first
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    # Open /dev/null and duplicate fds 1 and 2 to it
    with open(os.devnull, "w") as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            os.dup2(devnull.fileno(), 1)   # redirect fd1 -> /dev/null
            os.dup2(devnull.fileno(), 2)   # redirect fd2 -> /dev/null
            # Also silence Python-level streams for good measure
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            # Restore Python streams first
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # Restore original fds
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

# with contextlib.redirect_stdout(.StringIO()):
# with suppress_stdout():
with silence_stdio():
  exec(target_src, target_ctx)
  raise Exception("what about this?")


print("success!")

# def action():
#   originals = [Path("kernelbench_eval/original_kernel.py")] * 24
#   targets   = list(Path("kernelbench_eval/kernel_solutions").glob("*.py"))

#   print("printing ...")

#   out = parallel_eval_lists(
#     originals,
#     targets,
#     max_gpus=4,
#     runs=10,
#     seed=42,
#     print_progress=True,
#   )

# def main():
#   # import torch.multiprocessing as mp
#   # try:
#   #     mp.set_start_method("spawn", force=True)
#   # except RuntimeError:
#   #     pass
#   action()

# if __name__ == "__main__":
#   main()