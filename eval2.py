#!/usr/bin/env python3
"""
simple_kernelbench_eval.py

Minimal eval harness for KernelBench-style tasks:
- Correctness: nested torch.allclose (supports tensors, lists/tuples, dicts)
- Compliance: lightweight AST checks for banned imports/calls
- Runtime: latency over N independent runs with different seeds
           (re-seeds torch/cuda/numpy/random, regenerates inputs per run)

Assumptions:
- The reference dataset provides for each problem:
    Model(*get_init_inputs()), get_inputs(...)
- Candidate file defines:
    ModelNew(*get_init_inputs())
- Inputs for timing are taken from the ORIGINAL module's get_inputs, called
  under distinct RNG seeds; if get_inputs supports a `seed=` kwarg, we'll use it.

Example:
  python simple_kernelbench_eval.py \
    --dataset kernelbench \
    --glob "runs/**/*.py" \
    --device cuda:0 \
    --runtime-seeds 100 \
    --seed-base 0 \
    --per-seed-warmup 1 \
    --report kernelbench_eval.jsonl
"""

import argparse
import ast
import json
import random
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
import numpy as np
import torch
from datasets import load_dataset

FNAME_RE = re.compile(r"kernel_level_(\d+)_problem_(\d+)_sample_(\d+)\.py")


# -------------------------- Utilities --------------------------

def parse_filename(path: Path) -> Tuple[int, int, int]:
    m = FNAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Unrecognized filename format: {path.name}")
    level, pid, sid = map(int, m.groups())
    return level, pid, sid


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip()


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


# -------------------------- Compliance --------------------------

def compliance_check(code: str,
                     banned_imports: Iterable[str],
                     banned_calls: Iterable[str]) -> Tuple[bool, List[str]]:
    """
    Static checks:
      - Disallow specified 'import X' / 'from X import ...'
      - Disallow calls to specified names (supports dotted, e.g., time.sleep)
    """
    reasons: List[str] = []
    banned_imports = set(banned_imports)
    banned_calls = set(banned_calls)

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"SyntaxError: {e}"]

    class V(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in banned_imports:
                    reasons.append(f"Disallowed import: {alias.name}")
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module:
                root = node.module.split(".")[0]
                if root in banned_imports:
                    reasons.append(f"Disallowed import-from: {node.module}")
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                parts = []
                cur = node.func
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                    parts.reverse()
                    name = ".".join(parts)
            if name and name in banned_calls:
                reasons.append(f"Disallowed call: {name}()")
            self.generic_visit(node)

    V().visit(tree)
    return (len(reasons) == 0), reasons


# -------------------------- Seeding & Timing --------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def call_get_inputs_seeded(get_inputs, seed: int):
    """
    Seed RNGs then prefer get_inputs(seed=seed); fallback to get_inputs().
    """
    set_all_seeds(seed)
    try:
        return get_inputs(seed=seed)
    except TypeError:
        return get_inputs()


def _forward(model: torch.nn.Module, inputs: Any):
    return model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)


def time_over_seeds(model: torch.nn.Module,
                    get_inputs_fn,
                    device: torch.device,
                    seeds: List[int],
                    per_seed_warmup: int = 0) -> List[float]:
    """
    Returns per-seed latencies (ms). For each seed:
      - reseed RNGs
      - regenerate inputs
      - optional per-seed warmup
      - single timed forward
    """
    latencies_ms: List[float] = []
    use_cuda = (device.type == "cuda")

    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        for s in seeds:
            inputs = call_get_inputs_seeded(get_inputs_fn, s)
            inputs = to_device(inputs, device)

            # Optional per-seed warmup
            for _ in range(max(0, per_seed_warmup)):
                _ = _forward(model, inputs)
            if use_cuda:
                torch.cuda.synchronize()

            if use_cuda:
                starter.record()
                _ = _forward(model, inputs)
                ender.record()
                torch.cuda.synchronize()
                lat = starter.elapsed_time(ender)  # ms
            else:
                t0 = time.perf_counter()
                _ = _forward(model, inputs)
                t1 = time.perf_counter()
                lat = (t1 - t0) * 1000.0
            latencies_ms.append(float(lat))

    return latencies_ms


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": None, "std": None, "p50": None, "p95": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


# -------------------------- Dataset access --------------------------

def get_original_src(level: int,
                     pid: int,
                     cache: Dict[int, Dict[int, str]],
                     dataset_name: str) -> str:
    if level not in cache:
        split = f"level_{level}"
        ds = load_dataset(dataset_name, split=split)
        cache[level] = {int(r["problem_id"]): r["code"] for r in ds}
    try:
        return cache[level][pid]
    except KeyError:
        raise RuntimeError(f"Problem {pid} not found in level_{level} of dataset {dataset_name}")


# -------------------------- Core evaluation --------------------------

def evaluate_one(original_src: str,
                 candidate_src: str,
                 device: torch.device,
                 atol: float,
                 rtol: float,
                 warmup: int,
                 banned_imports: Iterable[str],
                 banned_calls: Iterable[str],
                 runtime_seeds: int = 100,
                 seed_base: int = 0,
                 per_seed_warmup: int = 0) -> Dict[str, Any]:
    """
    Returns a dict with:
      correct, compliant, compliance_reasons,
      rt_ref_stats, rt_new_stats, avg_ms_ref, avg_ms_new, speedup_x, error
    """
    result: Dict[str, Any] = {
        "correct": False,
        "compliant": False,
        "compliance_reasons": [],
        "rt_ref_stats": None,
        "rt_new_stats": None,
        "avg_ms_ref": None,
        "avg_ms_new": None,
        "speedup_x": None,
        "error": None,
    }

    # Compliance (static)
    compliant, reasons = compliance_check(candidate_src, banned_imports, banned_calls)
    result["compliant"] = compliant
    result["compliance_reasons"] = reasons

    # Exec original module
    try:
        octx: Dict[str, Any] = {"torch": torch}
        exec(original_src, octx)
        Model = octx.get("Model")
        get_init_inputs = octx.get("get_init_inputs")
        get_inputs = octx.get("get_inputs")
        if not callable(get_init_inputs) or not callable(get_inputs) or Model is None:
            raise RuntimeError("Original module must define Model, get_init_inputs, get_inputs")
    except Exception as e:
        result["error"] = f"OriginalExecError: {type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
        return result

    # Exec candidate module
    try:
        cctx: Dict[str, Any] = {"torch": torch}
        exec(candidate_src, cctx)
        ModelNew = cctx.get("ModelNew")
        if ModelNew is None:
            raise RuntimeError("Candidate must define ModelNew(*get_init_inputs())")
    except Exception as e:
        result["error"] = f"CandidateExecError: {type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
        return result

    # Build models and a single input (for correctness)
    try:
        init_inputs = get_init_inputs()
        model_ref = Model(*init_inputs).to(device).eval()
        model_new = ModelNew(*init_inputs).to(device).eval()

        raw_inputs = get_inputs()
        inputs = to_device(raw_inputs, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
    except Exception as e:
        result["error"] = f"BuildError: {type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
        return result

    # Correctness on one deterministic input
    try:
        with torch.inference_mode():
            out_ref = _forward(model_ref, inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            out_new = _forward(model_new, inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
        ok = allclose_nested(out_ref, out_new, atol=atol, rtol=rtol)
        result["correct"] = bool(ok)
        if not ok:
            result["error"] = f"Outputs differ beyond tolerance (atol={atol}, rtol={rtol})."
    except Exception as e:
        result["error"] = f"RunError: {type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
        return result

    # Global warmup once before per-seed timing (best-effort)
    try:
        with torch.inference_mode():
            for _ in range(max(0, warmup)):
                _ = _forward(model_ref, inputs)
                _ = _forward(model_new, inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass

    # Runtime over independent seeds
    try:
        seeds = [seed_base + i for i in range(runtime_seeds)]
        ref_latencies = time_over_seeds(model_ref, get_inputs, device, seeds, per_seed_warmup)
        new_latencies = time_over_seeds(model_new, get_inputs, device, seeds, per_seed_warmup)

        ref_stats = _summary_stats(ref_latencies)
        new_stats = _summary_stats(new_latencies)

        result["rt_ref_stats"] = {k: round(v, 4) if v is not None else None for k, v in ref_stats.items()}
        result["rt_new_stats"] = {k: round(v, 4) if v is not None else None for k, v in new_stats.items()}
        result["avg_ms_ref"] = result["rt_ref_stats"]["mean"]
        result["avg_ms_new"] = result["rt_new_stats"]["mean"]
        if result["avg_ms_ref"] is not None and result["avg_ms_new"] is not None and result["avg_ms_new"] > 0:
            result["speedup_x"] = round(result["avg_ms_ref"] / result["avg_ms_new"], 4)
    except Exception as e:
        result["error"] = f"TimingError: {type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"

    return result


# -------------------------- CLI / Orchestration --------------------------

def main():
    p = argparse.ArgumentParser(description="Simple KernelBench eval: correctness, runtime(over seeds), compliance.")
    p.add_argument("--dataset", type=str, default="kernelbench", help="HF dataset name")
    p.add_argument("--glob", type=str, required=True, help="Glob for candidate .py files")
    p.add_argument("--device", type=str, default="cuda:0", help="e.g., cuda:0 or cpu")
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=5, help="global warmup before timing")
    p.add_argument("--runtime-seeds", type=int, default=100, help="independent latency runs with different seeds")
    p.add_argument("--seed-base", type=int, default=0, help="starting seed; seeds are [base, base+N)")
    p.add_argument("--per-seed-warmup", type=int, default=0, help="warmup steps per seed before timing")
    p.add_argument("--report", type=str, default="", help="Write JSONL here if set")
    p.add_argument("--banned-imports", type=str,
                   default="subprocess,os,sys,pathlib,pickle,shutil",
                   help="Comma-separated root module names to ban")
    p.add_argument("--banned-calls", type=str,
                   default="open,eval,exec,__import__,time.sleep,torch.compile",
                   help="Comma-separated function names to ban (supports dotted)")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    files = sorted(Path(".").glob(args.glob))
    if not files:
        print(f"[WARN] No files matched: {args.glob}")
        return

    banned_imports = [s.strip() for s in args.banned_imports.split(",") if s.strip()]
    banned_calls = [s.strip() for s in args.banned_calls.split(",") if s.strip()]

    ds_cache: Dict[int, Dict[int, str]] = {}
    by_problem_pass: Dict[Tuple[int, int], bool] = {}

    report_f = open(args.report, "w", encoding="utf-8") if args.report else None

    total = 0
    passed_correctness = 0

    for f in tqdm(files, total=len(files)):
        total += 1
        try:
            level, pid, sid = parse_filename(f)
        except Exception as e:
            print(f"[SKIP] {f}: {e}")
            continue

        try:
            original_src = get_original_src(level, pid, ds_cache, args.dataset)
        except Exception as e:
            print(f"[FAIL] L{level} P{pid} S{sid} ({f.name}): could not load original ({e})")
            continue

        candidate_src = read_text(f)
        res = evaluate_one(
            original_src=original_src,
            candidate_src=candidate_src,
            device=device,
            atol=args.atol,
            rtol=args.rtol,
            warmup=args.warmup,
            banned_imports=banned_imports,
            banned_calls=banned_calls,
            runtime_seeds=args.runtime_seeds,
            seed_base=args.seed_base,
            per_seed_warmup=args.per_seed_warmup,
        )

        ok = res["correct"]
        if ok:
            passed_correctness += 1
            by_problem_pass[(level, pid)] = True

        # Pretty line
        comp = "COMPLIANT" if res["compliant"] else f"NONCOMPLIANT ({'; '.join(res['compliance_reasons'])})"
        rt = ""
        if res["avg_ms_ref"] is not None:
            rt = (f" | ref_mean={res['avg_ms_ref']}ms new_mean={res['avg_ms_new']}ms "
                  f"speedup={res['speedup_x']}x "
                  f"(ref p50={res['rt_ref_stats']['p50']} p95={res['rt_ref_stats']['p95']}; "
                  f"new p50={res['rt_new_stats']['p50']} p95={res['rt_new_stats']['p95']})")
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] L{level} P{pid} S{sid} ({f.name}) | {comp}{rt}")
        if res["error"] and not ok:
            print(f"        error: {res['error'].splitlines()[0]}")

        if report_f:
            row = {
                "level": level,
                "problem_id": pid,
                "sample_id": sid,
                "filename": f.name,
                **res,
            }
            report_f.write(json.dumps(row) + "\n")

    if report_f:
        report_f.close()
        print(f"[INFO] wrote report -> {args.report}")

    # Summary
    unique_problems = len({(parse_filename(f)[0], parse_filename(f)[1]) for f in files if FNAME_RE.search(f.name)})
    problems_solved = len(by_problem_pass)
    print("\n=== SUMMARY ===")
    print(f"Samples tested: {total}")
    print(f"Samples correct: {passed_correctness}")
    if unique_problems:
        print(f"Pass@k (per-problem): {problems_solved}/{unique_problems} = {problems_solved/unique_problems:.2%}")


if __name__ == "__main__":
    main()