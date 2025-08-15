import json
import os
import re
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import hydra
from omegaconf import OmegaConf
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from datasets import load_dataset
from src.utils import suppress_output_fds


FNAME_RE = re.compile(r"kernel_level_(\d+)_problem_(\d+)_sample_(\d+)\.py")


def parse_filename(path: Path) -> Tuple[int, int, int]:
    m = FNAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Unrecognized filename format: {path.name}")
    level, pid, sid = map(int, m.groups())
    return level, pid, sid

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

def load_kernel_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()

def build_problem_index(globbed_files: List[Path], dataset_name: str) -> Dict[Tuple[int, int], str]:
    """
    Build { (level, problem_id) -> original_code } once to avoid per-file HF dataset calls.
    """
    by_level: Dict[int, set] = {}
    for p in globbed_files:
        level, pid, _ = parse_filename(p)
        by_level.setdefault(level, set()).add(pid)

    index: Dict[Tuple[int, int], str] = {}
    for level, pids in by_level.items():
        split = f"level_{level}"
        ds = load_dataset(dataset_name, split=split)
        pid_to_code = {int(r["problem_id"]): r["code"] for r in ds}
        for pid in pids:
            if pid not in pid_to_code:
                raise RuntimeError(f"Problem id {pid} not found in split {split}")
            index[(level, pid)] = pid_to_code[pid]
    return index


def run_eval_core(
    *,
    original_src: str,
    candidate_src: str,
    device_id: int,
    atol: float,
    rtol: float,
) -> None:
    """
    Raises on mismatch/error; returns None on success.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this evaluation.")
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    # Exec original module
    orig_ctx: Dict[str, Any] = {}
    with suppress_output_fds():
        exec(original_src, orig_ctx)
    get_init_inputs = orig_ctx.get("get_init_inputs")
    get_inputs = orig_ctx.get("get_inputs")
    Model = orig_ctx.get("Model")
    if not callable(get_init_inputs) or not callable(get_inputs) or Model is None:
        raise RuntimeError("Original module missing get_init_inputs/get_inputs/Model")

    # Exec candidate module
    cand_ctx: Dict[str, Any] = {}
    with suppress_output_fds():
        exec(candidate_src, cand_ctx)
    ModelNew = cand_ctx.get("ModelNew")
    if ModelNew is None:
        raise RuntimeError("Candidate module missing ModelNew")

    # Build models & compare
    init_inputs = to_device(get_init_inputs(), device)
    with torch.inference_mode():
        ref = Model(*init_inputs).to(device).eval()
        new = ModelNew(*init_inputs).to(device).eval()
        inputs = to_device(get_inputs(), device)

        out_ref = ref(*inputs)
        torch.cuda.synchronize(device=device)
        out_new = new(*inputs)
        torch.cuda.synchronize(device=device)

        if not allclose_nested(out_ref, out_new, atol=atol, rtol=rtol):
            raise AssertionError(f"Outputs differ beyond tolerance (atol={atol}, rtol={rtol}).")


@dataclass
class EvalResult:
    level: int
    problem_id: int
    sample_id: int
    status: str             # "pass" | "fail" | "skipped"
    error: Optional[str] = None
    filename: Optional[str] = None


def worker_task(
    file_path: str,
    original_src: str,
    device_id: int,
    atol: float,
    rtol: float,
    solved_key: Tuple[int, int],
    solved_dict: Dict[Tuple[int, int], bool],
    lock: mp.Lock,
) -> EvalResult:
    path = Path(file_path)
    level, pid, sid = parse_filename(path)

    # Early skip if already solved
    with lock:
        if solved_dict.get(solved_key, False):
            return EvalResult(level, pid, sid, status="skipped", filename=path.name)

    try:
        candidate_src = load_kernel_text(path)
        run_eval_core(
            original_src=original_src,
            candidate_src=candidate_src,
            device_id=device_id,
            atol=atol,
            rtol=rtol,
        )
        with lock:
            solved_dict[solved_key] = True
        return EvalResult(level, pid, sid, status="pass", filename=path.name)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc(limit=2)
        return EvalResult(level, pid, sid, status="fail", error=f"{err}\n{tb}", filename=path.name)


def evaluate_files(
    files: List[Path],
    dataset_name: str,
    atol: float,
    rtol: float,
    gpu_ids: List[int],
    workers_per_gpu: int,
) -> List[EvalResult]:
    # Build original-code index once
    idx = build_problem_index(files, dataset_name)

    # Distribute tasks round-robin over gpu_ids
    tasks: List[Tuple[str, str, int, float, float, Tuple[int, int]]] = []
    for i, f in enumerate(files):
        level, pid, sid = parse_filename(f)
        original_src = idx[(level, pid)]
        device_id = gpu_ids[i % len(gpu_ids)]
        tasks.append((str(f), original_src, device_id, atol, rtol, (level, pid)))

    manager = mp.Manager()
    solved = manager.dict()  # (level,pid) -> bool
    lock = manager.Lock()

    results: List[EvalResult] = []

    # Create one executor per GPU (keeps CUDA contexts isolated & simple)
    executors = []
    per_gpu_chunks: Dict[int, List[Tuple]] = {gpu: [] for gpu in gpu_ids}
    for t in tasks:
        per_gpu_chunks[t[2]].append(t)

    for gpu in gpu_ids:
        ex = ProcessPoolExecutor(
            max_workers=workers_per_gpu,
            mp_context=mp.get_context("spawn"),
        )
        executors.append((gpu, ex))

    # Submit per-GPU
    futures = []
    for gpu, ex in executors:
        for (file_path, original_src, device_id, a, r, solved_key) in per_gpu_chunks[gpu]:
            fut = ex.submit(worker_task, file_path, original_src, device_id, a, r, solved_key, solved, lock)
            futures.append(fut)

    # Collect
    for fut in futures:
        res: EvalResult = fut.result()
        results.append(res)

    # Cleanup
    for _, ex in executors:
        ex.shutdown(wait=True)

    return results


def summarize(results: List[EvalResult]) -> None:
    # pass@k per problem (pass if any sample passed)
    by_problem: Dict[Tuple[int, int], List[EvalResult]] = {}
    for r in results:
        by_problem.setdefault((r.level, r.problem_id), []).append(r)

    total = len(by_problem)
    passed = sum(1 for _k, rs in by_problem.items() if any(r.status == "pass" for r in rs))

    print("\n=== RESULTS ===")
    print(f"Total problems: {total}")
    print(f"Passed problems: {passed}")
    if total:
        print(f"Pass@k accuracy: {passed}/{total} = {passed/total:.2%}")

    # Per-problem line
    for (lvl, pid) in sorted(by_problem.keys()):
        rs = by_problem[(lvl, pid)]
        status = "PASS" if any(r.status == "pass" for r in rs) else "FAIL"
        tested = sum(1 for r in rs if r.status != "skipped")
        print(f"Level {lvl} Problem {pid}: {status} (tested {tested} samples)")


def maybe_write_report(results: List[EvalResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"[INFO] Wrote report: {report_path}")


@hydra.main(config_path="conf", config_name="eval", version_base=None)
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    glob_dir = Path(cfg.io.glob_dir)
    files = sorted(glob_dir.glob(cfg.io.glob_pattern))
    if not files:
        print(f"[WARNING] No files found under {glob_dir} matching {cfg.io.glob_pattern}")
        return

    # GPU selection
    if cfg.parallel.gpus is None:
        num = torch.cuda.device_count()
        if num < 1:
            raise RuntimeError("CUDA is required; no GPUs detected.")
        gpu_ids = list(range(num))
    else:
        gpu_ids = list(cfg.parallel.gpus)

    workers_per_gpu = int(cfg.parallel.workers_per_gpu)
    if not cfg.parallel.enable:
        gpu_ids = [gpu_ids[0]]
        workers_per_gpu = 1

    # Run
    results = evaluate_files(
        files=files,
        dataset_name=cfg.dataset.name,
        atol=float(cfg.tolerance.atol),
        rtol=float(cfg.tolerance.rtol),
        gpu_ids=gpu_ids,
        workers_per_gpu=workers_per_gpu,
    )

    summarize(results)

    if cfg.io.save_report:
        maybe_write_report(results, Path(cfg.io.report_file))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()