"""
Remote kernel evaluation via Modal.

Offloads CUDA kernel compilation + profiling to Modal's GPU cloud,
freeing local GPUs for training and vLLM.
"""
import time
from dataclasses import asdict
from typing import List, Optional, Sequence, Union
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app & image
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential", "python3-dev", "ninja-build", "gcc-11", "g++-11")
    .pip_install(
        "torch==2.7.1",
        "numpy==2.2.6",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
)

app = modal.App("kernelbench-eval", image=image)

# ---------------------------------------------------------------------------
# Remote evaluation function (runs on Modal GPU)
# ---------------------------------------------------------------------------
@app.function(gpu="L4", timeout=600, max_containers=100, scaledown_window=2)
@modal.concurrent(max_inputs=1)
def eval_kernel(
    index: int,
    original_src: str,
    target_src: str,
    num_perf_runs: int = 1,
    seed: int = 42,
) -> dict:
    """Evaluate a single (original, target) kernel pair on a remote GPU."""
    import os, gc, time, types, tempfile
    import torch

    device = torch.device("cuda:0")

    # Reset CUDA context to clear any corruption from previous runs
    torch.cuda.empty_cache()
    gc.collect()
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass

    # Seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Set SM arch for this GPU
    maj, minr = torch.cuda.get_device_capability(0)
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{maj}.{minr}"

    # Unique build cache
    cache_dir = os.path.join(tempfile.gettempdir(), f"torch_extensions_{os.getpid()}")
    os.environ["TORCH_EXTENSIONS_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # Shim for common import typo
    try:
        import torch.utils.cpp_extension as cpp_extension
        mod = types.ModuleType("torch.utils.cppextension")
        mod.__dict__.update(cpp_extension.__dict__)
        import sys
        sys.modules["torch.utils.cppextension"] = mod
        import torch.utils as torch_utils
        setattr(torch_utils, "cppextension", mod)
    except Exception:
        pass

    result = {
        "index": index,
        "device_id": 0,
        "is_compiled": False,
        "is_executed": False,
        "is_correct": False,
        "avg_speed_up": None,
        "median_speed_up": None,
        "speed_ups": None,
        "error": None,
        "worker_ms": None,
    }

    t0 = time.perf_counter()
    try:
        # Compile original
        orig_ctx = {}
        try:
            compile(original_src, "<string>", "exec")
        except Exception as e:
            raise _CompilationError(f"Error compiling original kernel: {e}")
        try:
            exec(original_src, orig_ctx)
        except Exception as e:
            raise _ExecutionError(f"Error executing original kernel: {e}")

        # Compile target
        tgt_ctx = {}
        try:
            compile(target_src, "<string>", "exec")
        except Exception as e:
            raise _CompilationError(f"Error compiling target kernel: {e}")
        try:
            exec(target_src, tgt_ctx)
        except Exception as e:
            raise _ExecutionError(f"Error executing target kernel: {e}")

        # Profile
        try:
            speed_ups = _profile(orig_ctx, tgt_ctx, device, num_perf_runs=num_perf_runs)
        except _OutputMismatchError as e:
            raise _OutputMismatchError(str(e))
        except Exception as e:
            raise _ExecutionError(f"Error profiling target kernel: {e}...")

        su_sorted = sorted(speed_ups)
        result["speed_ups"] = [float(x) for x in speed_ups]
        result["avg_speed_up"] = sum(speed_ups) / len(speed_ups)
        result["median_speed_up"] = float(su_sorted[len(su_sorted) // 2])
        result["is_compiled"] = True
        result["is_executed"] = True
        result["is_correct"] = True

    except _CompilationError as e:
        result["error"] = f"CompilationError: {e}"
    except _ExecutionError as e:
        result["is_compiled"] = True
        result["error"] = f"ExecutionError: {e}"
    except _OutputMismatchError as e:
        result["is_compiled"] = True
        result["is_executed"] = True
        result["error"] = f"OutputMismatchError: {e}"
    except Exception as e:
        import traceback
        result["error"] = f"Unhandled {type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        try:
            torch.cuda.synchronize(device=device)
        except Exception:
            pass
        result["worker_ms"] = (time.perf_counter() - t0) * 1000.0
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Helper exceptions & profiling (inlined to avoid import issues on Modal)
# ---------------------------------------------------------------------------
class _CompilationError(Exception):
    pass

class _ExecutionError(Exception):
    pass

class _OutputMismatchError(Exception):
    pass


def _to_device(x, device):
    import torch
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


def _allclose_nested(a, b, *, atol, rtol):
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return False
        try:
            return torch.allclose(a, b.to(a.dtype), atol=atol, rtol=rtol)
        except Exception:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        return all(_allclose_nested(x, y, atol=atol, rtol=rtol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return all(_allclose_nested(a[k], b[k], atol=atol, rtol=rtol) for k in a.keys())
    return a == b


def _profile_kernel(ctx, ctx_key, inputs, init_inputs, device, warmup_runs=1, num_perf_runs=1):
    import torch
    Model = ctx.get(ctx_key)
    if Model is None:
        raise RuntimeError(f"Candidate module missing {ctx_key}")
    with torch.inference_mode():
        ref = Model(*init_inputs).to(device).eval()
        inputs = _to_device(inputs, device)
        for _ in range(warmup_runs):
            _ = ref(*inputs)
        durations = []
        out = None
        for _ in range(num_perf_runs):
            start = time.perf_counter()
            out = ref(*inputs)
            torch.cuda.synchronize(device=device)
            durations.append(time.perf_counter() - start)
    return out, durations


def _profile(ctx, target_ctx, device, atol=1e-2, rtol=1e-2, num_perf_runs=1):
    Model = ctx.get("Model")
    get_inputs = ctx.get("get_inputs")
    get_init_inputs = ctx.get("get_init_inputs")
    inputs = get_inputs()
    init_inputs = get_init_inputs()
    res_ref, duration_refs = _profile_kernel(ctx, "Model", inputs, init_inputs, device)
    res_tgt, durations_tgt = _profile_kernel(target_ctx, "ModelNew", inputs, init_inputs, device, num_perf_runs=num_perf_runs)
    ok = _allclose_nested(res_ref, res_tgt, atol=atol, rtol=rtol)
    if not ok:
        raise _OutputMismatchError(f"Outputs differ beyond tolerance: (atol={atol}, rtol={rtol})")
    assert len(duration_refs) == 1
    return [duration_refs[0] / d for d in durations_tgt]


# ---------------------------------------------------------------------------
# Public API: drop-in replacement for parallel_eval_lists
# ---------------------------------------------------------------------------
def modal_eval_lists(
    original_srcs: Sequence[Union[str, Path]],
    target_srcs: Sequence[Union[str, Path]],
    *,
    runs: int = 1,
    seed: int = 42,
    timeout: Optional[float] = None,
    print_progress: bool = True,
) -> list:
    """
    Evaluate (original, target) pairs on Modal GPUs.
    Drop-in replacement for parallel_eval_lists.

    Each pair is dispatched via fn.remote() in a separate thread,
    so all containers run in parallel. Results are collected as
    each thread completes.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.kernelbench_eval.run_parallel import KernelEvalResult

    if len(original_srcs) != len(target_srcs):
        raise ValueError(f"Length mismatch: {len(original_srcs)} vs {len(target_srcs)}")

    n = len(original_srcs)
    fn = modal.Function.from_name("kernelbench-eval", "eval_kernel")
    t0 = time.time()
    done = 0

    def _call_remote(i):
        original_code = Path(original_srcs[i]).read_text()
        target_code = Path(target_srcs[i]).read_text()
        return fn.remote(i, original_code, target_code, runs, seed)

    results: List[Optional[KernelEvalResult]] = [None] * n

    with ThreadPoolExecutor(max_workers=n) as pool:
        future_to_idx = {pool.submit(_call_remote, i): i for i in range(n)}

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                raw = future.result()
                r = KernelEvalResult(
                    index=raw["index"],
                    original=str(original_srcs[raw["index"]]),
                    target=str(target_srcs[raw["index"]]),
                    device_id=raw["device_id"],
                    is_compiled=raw["is_compiled"],
                    is_executed=raw["is_executed"],
                    is_correct=raw["is_correct"],
                    avg_speed_up=raw.get("avg_speed_up"),
                    median_speed_up=raw.get("median_speed_up"),
                    speed_ups=raw.get("speed_ups"),
                    error=raw.get("error"),
                    worker_ms=raw.get("worker_ms"),
                    submit_to_get_ms=(time.time() - t0) * 1000.0,
                )
            except Exception as e:
                r = KernelEvalResult(
                    index=idx,
                    original=str(original_srcs[idx]),
                    target=str(target_srcs[idx]),
                    device_id=-1,
                    error=f"ModalError: {type(e).__name__}: {e}",
                )

            results[idx] = r
            done += 1

            if print_progress:
                status = "correct" if r.is_correct else ("executed" if r.is_executed else ("compiled" if r.is_compiled else "failed"))
                err = f" | error={r.error[:80]}" if r.error else ""
                print(f"[modal] [{r.index:02d}] {status} ({done}/{n}){err}")

    return results
