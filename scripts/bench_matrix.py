# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "ninja",
#     "numpy",
# ]
# ///
"""Release benchmark matrix for CUDA training and inference.

This is the README-number generator. It benchmarks the three public V2
surfaces against naive PyTorch baselines and prints copy-paste Markdown
tables. The headline section is contrastive training because that is the
real fine-tuning story for ColBERT-style models.

Triggered via::

    just cuda-bench-matrix              # a100-large default
    just cuda-bench-matrix a100x4
    just cuda-bench-matrix l40sx1
    just bench-matrix-metal             # Apple Silicon / MPS
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEVICE = "cuda"


def _ensure(pkg: str) -> None:
    try:
        __import__(pkg)
        return
    except ImportError:
        pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])


def _setup_bridge(repo: Path):
    import torch
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    sm = f"{major}{minor}"
    ext = load(
        name="maxsim_cuda_dev",
        sources=[
            str(repo / "maxsim_cuda" / "maxsim.cu"),
            str(repo / "maxsim_cuda" / "dev_binding.cpp"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            f"-gencode=arch=compute_{sm},code=sm_{sm}",
        ],
        extra_cflags=["-O3"],
        verbose=False,
    )
    sys.path.insert(0, str(repo / "torch-ext"))
    fake_ops = types.ModuleType("maxsim._ops")
    fake_ops.ops = types.SimpleNamespace(
        maxsim_forward=ext.maxsim_forward,
        maxsim_padded_forward=ext.maxsim_padded_forward,
        maxsim_padded_forward_with_argmax=ext.maxsim_padded_forward_with_argmax,
        maxsim_padded_backward=ext.maxsim_padded_backward,
        maxsim_packed_forward_with_argmax=ext.maxsim_packed_forward_with_argmax,
        maxsim_packed_backward=ext.maxsim_packed_backward,
        maxsim_contrastive_forward=ext.maxsim_contrastive_forward,
        maxsim_contrastive_forward_with_argmax=ext.maxsim_contrastive_forward_with_argmax,
        maxsim_contrastive_backward=ext.maxsim_contrastive_backward,
    )
    sys.modules["maxsim._ops"] = fake_ops


def _setup_metal_import(repo: Path) -> None:
    import torch

    major_minor = ".".join(torch.__version__.split(".")[:2])
    torch_tag = "torch" + major_minor.replace(".", "")
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        path = Path(entry)
        if (
            path.exists()
            and path.name.startswith(torch_tag)
            and (path / "_ops.py").exists()
        ):
            sys.path.insert(0, str(path))
            return
    for dirname in ("build", "result"):
        build_dir = repo / dirname
        if not build_dir.exists():
            continue
        candidates = sorted(build_dir.glob(f"{torch_tag}-*")) or sorted(
            build_dir.glob("torch*")
        )
        if candidates:
            sys.path.insert(0, str(candidates[-1]))
            return
    raise RuntimeError(
        "no Metal maxsim build found. Run `just build` before `just bench-matrix-metal`."
    )


@dataclass(frozen=True)
class Result:
    surface: str
    preset: str
    description: str
    shape: dict
    dtype: str
    kernel_ms: float | None
    naive_ms: float | None
    kernel_fwd_ms: float | None
    naive_fwd_ms: float | None
    padded_ms: float | None
    kernel_peak_mb: float | None
    naive_peak_mb: float | None
    retained_reduction: float | None
    correct: bool
    note: str = ""

    @property
    def speedup(self) -> float | None:
        if self.kernel_ms is None or self.naive_ms is None or self.kernel_ms == 0:
            return None
        return self.naive_ms / self.kernel_ms

    @property
    def kernel_bwd_ms(self) -> float | None:
        if self.kernel_ms is None or self.kernel_fwd_ms is None:
            return None
        return max(self.kernel_ms - self.kernel_fwd_ms, 0.0)

    @property
    def naive_bwd_ms(self) -> float | None:
        if self.naive_ms is None or self.naive_fwd_ms is None:
            return None
        return max(self.naive_ms - self.naive_fwd_ms, 0.0)

    @property
    def bwd_speedup(self) -> float | None:
        if self.kernel_bwd_ms in (None, 0.0) or self.naive_bwd_ms is None:
            return None
        return self.naive_bwd_ms / self.kernel_bwd_ms

    @property
    def peak_ratio(self) -> float | None:
        if self.kernel_peak_mb in (None, 0.0) or self.naive_peak_mb is None:
            return None
        return self.naive_peak_mb / self.kernel_peak_mb

    def to_dict(self) -> dict:
        return {
            "surface": self.surface,
            "preset": self.preset,
            "description": self.description,
            "shape": self.shape,
            "dtype": self.dtype,
            "maxsim_step_ms": self.kernel_ms,
            "naive_step_ms": self.naive_ms,
            "step_speedup": self.speedup,
            "maxsim_fwd_ms": self.kernel_fwd_ms,
            "naive_fwd_ms": self.naive_fwd_ms,
            "padded_ms": self.padded_ms,
            "maxsim_bwd_ms": self.kernel_bwd_ms,
            "naive_bwd_ms": self.naive_bwd_ms,
            "bwd_speedup": self.bwd_speedup,
            "maxsim_peak_mb": self.kernel_peak_mb,
            "naive_peak_mb": self.naive_peak_mb,
            "peak_ratio": self.peak_ratio,
            "retained_reduction": self.retained_reduction,
            "correct": self.correct,
            "note": self.note,
        }


def _fmt_ms(v: float | None) -> str:
    return "OOM" if v is None else f"{v:.3f}"


def _fmt_x(v: float | None) -> str:
    return "OOM" if v is None else f"{v:.2f}x"


def _fmt_mb(v: float | None) -> str:
    return "OOM" if v is None else f"{v:.1f}"


def _check_grads(name: str, got, expected, *, rtol: float, atol: float) -> tuple[bool, str]:
    import torch

    got_f = got.float()
    exp_f = expected.float()
    ok = torch.allclose(got_f, exp_f, rtol=rtol, atol=atol)
    diff = (got_f - exp_f).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = exp_f.abs().clamp_min(atol)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    return ok, f"{name}: max_abs={max_abs:.4g}, max_rel={max_rel:.4g}"


def _sync() -> None:
    import torch

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def _clear_cache() -> None:
    import torch

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


def _is_oom(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


def _time_step(fn: Callable[[], object], n_iter: int, n_warmup: int) -> float | None:
    import torch

    try:
        for _ in range(n_warmup):
            fn()
        _sync()
        times_ms: list[float] = []
        if DEVICE == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(n_iter):
                start.record()
                fn()
                end.record()
                _sync()
                times_ms.append(start.elapsed_time(end))
        else:
            for _ in range(n_iter):
                t0 = time.perf_counter()
                fn()
                _sync()
                times_ms.append((time.perf_counter() - t0) * 1000)
        return statistics.median(times_ms)
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        _clear_cache()
        return None


def _peak_memory_mb(fn: Callable[[], object]) -> float | None:
    import torch

    if DEVICE != "cuda":
        return None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        fn()
        _sync()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def _dtype_label(dtype) -> str:
    import torch

    return "fp16" if dtype == torch.float16 else "bf16"


def _make_contrastive(Nq, Nb, Lq, Ld, dim, dtype, device, seed):
    import torch

    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(Nq, Lq, dim, generator=gen, dtype=dtype).to(device)
    d_lens = [Ld] * Nb
    docs = torch.randn(Nb * Ld, dim, generator=gen, dtype=dtype).to(device)
    cu = torch.arange(0, (Nb + 1) * Ld, Ld, dtype=torch.int32, device=device)
    return q, docs, cu, d_lens


def _naive_contrastive(q, docs, cu, d_lens):
    import torch

    Nq, _Lq, dim = q.shape
    Nb = len(d_lens)
    Ld_max = max(d_lens)
    docs_padded = docs.new_zeros((Nb, Ld_max, dim))
    offs = cu.to(torch.int64).cpu().tolist()
    for i, ld_i in enumerate(d_lens):
        docs_padded[i, :ld_i] = docs[offs[i] : offs[i + 1]]
    sim = torch.einsum("qid,njd->qnij", q.float(), docs_padded.float())
    return sim.max(dim=-1).values.sum(dim=-1)


def _bench_contrastive(maxsim, name, shape, dtype, n_iter, n_warmup) -> Result:
    import torch

    q, docs, cu, d_lens = _make_contrastive(
        **shape, dtype=dtype, device=DEVICE, seed=1234
    )

    def k_step():
        q_r = q.detach().clone().requires_grad_(True)
        d_r = docs.detach().clone().requires_grad_(True)
        maxsim.score_contrastive_train(q_r, d_r, cu).sum().backward()

    def n_step():
        q_r = q.detach().clone().requires_grad_(True)
        d_r = docs.detach().clone().requires_grad_(True)
        _naive_contrastive(q_r, d_r, cu, d_lens).sum().backward()

    def k_fwd():
        q_r = q.detach().clone().requires_grad_(True)
        d_r = docs.detach().clone().requires_grad_(True)
        return maxsim.score_contrastive_train(q_r, d_r, cu)

    def n_fwd():
        q_r = q.detach().clone().requires_grad_(True)
        d_r = docs.detach().clone().requires_grad_(True)
        return _naive_contrastive(q_r, d_r, cu, d_lens)

    correct = True
    note = ""
    try:
        q_k = q.detach().clone().requires_grad_(True)
        d_k = docs.detach().clone().requires_grad_(True)
        q_n = q.detach().clone().requires_grad_(True)
        d_n = docs.detach().clone().requires_grad_(True)
        maxsim.score_contrastive_train(q_k, d_k, cu).sum().backward()
        _naive_contrastive(q_n, d_n, cu, d_lens).sum().backward()
        q_ok, q_note = _check_grads("dq", q_k.grad, q_n.grad, rtol=3e-2, atol=3e-2)
        d_ok, d_note = _check_grads("dd", d_k.grad, d_n.grad, rtol=3e-2, atol=3e-2)
        correct = q_ok and d_ok
        note = "" if correct else f"{q_note}; {d_note}"
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        correct = False
        note = "correctness check OOM"
        _clear_cache()

    Ld = max(d_lens)
    reduction = float(Ld)
    structured_shape = {
        "Nq": shape["Nq"],
        "Nb": shape["Nb"],
        "Lq": shape["Lq"],
        "Ld": Ld,
        "D": shape["dim"],
    }
    return Result(
        surface="contrastive_train",
        preset=name,
        description=CONTRASTIVE_DESCRIPTIONS.get(name, ""),
        shape=structured_shape,
        dtype=_dtype_label(dtype),
        kernel_ms=_time_step(k_step, n_iter, n_warmup),
        naive_ms=_time_step(n_step, n_iter, n_warmup),
        kernel_fwd_ms=_time_step(k_fwd, n_iter, n_warmup),
        naive_fwd_ms=_time_step(n_fwd, n_iter, n_warmup),
        padded_ms=None,
        kernel_peak_mb=_peak_memory_mb(k_step),
        naive_peak_mb=_peak_memory_mb(n_step),
        retained_reduction=reduction,
        correct=correct,
        note=note,
    )


def _make_padded(B, C, Lq, Ld, dim, dtype, device, seed):
    import torch

    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, Lq, dim, generator=gen, dtype=dtype).to(device)
    d = torch.randn(B, C, Ld, dim, generator=gen, dtype=dtype).to(device)
    qlen = torch.full((B,), Lq, dtype=torch.int32, device=device)
    dlen = torch.full((B, C), Ld, dtype=torch.int32, device=device)
    return q, d, qlen, dlen


def _naive_padded(q, d, qlen, dlen):
    import torch

    _B, _C, Ld, _dim = d.shape
    Lq = q.shape[1]
    sim = torch.einsum("bid,bcjd->bcij", q.float(), d.float())
    q_mask = torch.arange(Lq, device=q.device)[None, :] < qlen[:, None]
    d_mask = torch.arange(Ld, device=q.device)[None, None, :] < dlen[:, :, None]
    sim = sim.masked_fill(~d_mask[:, :, None, :], float("-inf"))
    per_q_max = sim.max(dim=-1).values
    per_q_max = per_q_max.masked_fill(~q_mask[:, None, :], 0.0)
    return per_q_max.sum(dim=-1)


def _bench_padded_infer(maxsim, name, shape, dtype, n_iter, n_warmup) -> Result:
    import torch

    q, d, qlen, dlen = _make_padded(**shape, dtype=dtype, device=DEVICE, seed=1234)

    def k_step():
        return maxsim.score_candidates_padded(q, d, qlen, dlen)

    def n_step():
        return _naive_padded(q, d, qlen, dlen)

    correct = True
    note = ""
    try:
        scores_k = k_step()
        scores_n = n_step()
        correct = torch.allclose(scores_k.float(), scores_n.float(), rtol=3e-2, atol=3e-2)
        note = "" if correct else "scores differ from PyTorch einsum baseline"
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        correct = False
        note = "correctness check OOM"
        _clear_cache()

    structured_shape = {
        "B": shape["B"],
        "K": shape["C"],
        "Lq": shape["Lq"],
        "Ld": shape["Ld"],
        "D": shape["dim"],
    }
    return Result(
        surface="padded_infer",
        preset=name,
        description=PADDED_DESCRIPTIONS.get(name, ""),
        shape=structured_shape,
        dtype=_dtype_label(dtype),
        kernel_ms=_time_step(k_step, n_iter, n_warmup),
        naive_ms=_time_step(n_step, n_iter, n_warmup),
        kernel_fwd_ms=None,
        naive_fwd_ms=None,
        padded_ms=None,
        kernel_peak_mb=_peak_memory_mb(k_step),
        naive_peak_mb=_peak_memory_mb(n_step),
        retained_reduction=None,
        correct=correct,
        note=note,
    )


def _make_packed_from_padded(q, d):
    import torch

    B, Lq, D = q.shape
    Bd, K, Ld, Dd = d.shape
    assert B == Bd and D == Dd
    q_flat = q.reshape(B * Lq, D).contiguous()
    d_flat = d.reshape(B * K * Ld, D).contiguous()
    qoff = torch.arange(0, (B + 1) * Lq, Lq, dtype=torch.int32, device=q.device)
    doff = torch.arange(0, (B * K + 1) * Ld, Ld, dtype=torch.int32, device=q.device)
    pair_ids = torch.arange(B * K, dtype=torch.int32, device=q.device)
    qids = pair_ids // K
    dids = pair_ids
    return q_flat, qoff, d_flat, doff, qids, dids


def _bench_packed_infer(maxsim, name, shape, dtype, n_iter, n_warmup) -> Result:
    import torch

    q, d, qlen, dlen = _make_padded(**shape, dtype=dtype, device=DEVICE, seed=1234)
    q_flat, qoff, d_flat, doff, qids, dids = _make_packed_from_padded(q, d)

    def k_step():
        return maxsim.score_pairs_packed(
            q_flat, qoff, d_flat, doff, qids, dids, max_q_len=shape["Lq"]
        )

    def n_step():
        return _naive_padded(q, d, qlen, dlen)

    def padded_step():
        return maxsim.score_candidates_padded(q, d, qlen, dlen)

    correct = True
    note = ""
    try:
        scores_k = k_step().view(shape["B"], shape["C"])
        scores_p = padded_step()
        scores_n = n_step()
        correct = (
            torch.allclose(scores_k.float(), scores_p.float(), rtol=3e-2, atol=3e-2)
            and torch.allclose(scores_k.float(), scores_n.float(), rtol=3e-2, atol=3e-2)
        )
        note = "" if correct else "packed scores differ from padded/PyTorch baselines"
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        correct = False
        note = "correctness check OOM"
        _clear_cache()

    structured_shape = {
        "B": shape["B"],
        "K": shape["C"],
        "Lq": shape["Lq"],
        "Ld": shape["Ld"],
        "D": shape["dim"],
    }
    return Result(
        surface="packed_infer",
        preset=name,
        description=PACKED_DESCRIPTIONS.get(name, ""),
        shape=structured_shape,
        dtype=_dtype_label(dtype),
        kernel_ms=_time_step(k_step, n_iter, n_warmup),
        naive_ms=_time_step(n_step, n_iter, n_warmup),
        kernel_fwd_ms=None,
        naive_fwd_ms=None,
        padded_ms=_time_step(padded_step, n_iter, n_warmup),
        kernel_peak_mb=_peak_memory_mb(k_step),
        naive_peak_mb=_peak_memory_mb(n_step),
        retained_reduction=None,
        correct=correct,
        note=note,
    )


CONTRASTIVE_DESCRIPTIONS = {
    "Contrastive": (
        "Standard ColBERT-style in-batch contrastive shape: "
        "Nq=Nb=32, Lq=32, Ld=80, dim=128."
    ),
    "LongDocs": (
        "Same in-batch contrastive batch shape but with long docs "
        "(Ld=512) — stresses the retained-similarity tensor."
    ),
    "BigBatch": (
        "Doubled in-batch batch size (Nq=Nb=64) at Ld=128. Bigger grid; "
        "more SM-occupancy headroom."
    ),
}

PADDED_DESCRIPTIONS = {
    "Rerank": (
        "Padded reranking at typical inference shape: B=32 queries × "
        "K=50 candidates, Lq=32, Ld=180, dim=128."
    ),
    "HeavyRerank": (
        "Padded reranking at K=100 candidates per query — bigger memory "
        "and compute envelope."
    ),
}

PACKED_DESCRIPTIONS = {
    "PackedRerank": (
        "Same fixed-K reranking shape as Rerank, expressed through the packed "
        "pair API to measure packing/layout overhead."
    ),
    "PackedHeavyRerank": (
        "Same fixed-K reranking shape as HeavyRerank, expressed through the "
        "packed pair API."
    ),
}


def _slugify_gpu(gpu_name: str) -> str:
    """`NVIDIA A100-SXM4-80GB` → `a100-sxm4-80gb`. Stable, provider-agnostic.

    Drops vendor prefix, lowercases, collapses non-alnum to `-`.
    """
    import re

    name = gpu_name.lower()
    # Strip common vendor prefixes so we don't leak NVIDIA branding into the
    # filename for what is fundamentally a GPU-class identifier.
    for prefix in ("nvidia ", "amd ", "intel "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return name or "unknown-gpu"


def _summarise_repeats(runs: list[list[Result]]) -> tuple[list[Result], list[dict]]:
    """Reduce N runs of the same matrix into (median Result, variance dict)
    pairs. Median timings; min/max/spread captured per-workload for JSON."""
    import statistics as st

    summary: list[Result] = []
    variance: list[dict] = []
    n_workloads = len(runs[0])
    for w in range(n_workloads):
        per_run = [r[w] for r in runs]
        base = per_run[0]

        def _med(field: str) -> float | None:
            vals = [getattr(r, field) for r in per_run
                    if getattr(r, field) is not None]
            return st.median(vals) if vals else None

        merged = Result(
            surface=base.surface,
            preset=base.preset,
            description=base.description,
            shape=base.shape,
            dtype=base.dtype,
            kernel_ms=_med("kernel_ms"),
            naive_ms=_med("naive_ms"),
            kernel_fwd_ms=_med("kernel_fwd_ms"),
            naive_fwd_ms=_med("naive_fwd_ms"),
            padded_ms=_med("padded_ms"),
            kernel_peak_mb=_med("kernel_peak_mb"),
            naive_peak_mb=_med("naive_peak_mb"),
            retained_reduction=base.retained_reduction,
            correct=all(r.correct for r in per_run),
            note=base.note,
        )
        summary.append(merged)

        speedups = [r.speedup for r in per_run if r.speedup is not None]
        if len(speedups) >= 2:
            variance.append({
                "step_speedup_min": min(speedups),
                "step_speedup_max": max(speedups),
                "step_speedup_spread": max(speedups) - min(speedups),
                "step_speedup_stdev": st.stdev(speedups),
                "n_repeats": len(speedups),
            })
        else:
            variance.append({"n_repeats": len(speedups)})
    return summary, variance


def main() -> int:
    import datetime as dt

    global DEVICE

    repo = Path(os.environ.get("MAXSIM_BENCH_REPO", "/kernels/maxsim"))
    if not repo.is_dir():
        repo = Path.cwd()
    os.chdir(repo)

    import torch

    metal_mode = os.environ.get("MAXSIM_BENCH_METAL", "false").lower() in {
        "1", "true", "yes",
    }
    if metal_mode:
        if not torch.backends.mps.is_available():
            print("ERROR: `MAXSIM_BENCH_METAL=true` requires MPS.", file=sys.stderr)
            return 1
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        print("ERROR: maxsim benchmarks need CUDA or MPS.", file=sys.stderr)
        return 1

    n_iter = int(os.environ.get("MAXSIM_BENCH_ITERS", "30"))
    n_warmup = int(os.environ.get("MAXSIM_BENCH_WARMUP", "5"))
    n_repeats = int(os.environ.get("MAXSIM_BENCH_REPEATS", "1"))
    git_commit = os.environ.get("MAXSIM_BENCH_COMMIT", "unknown")
    git_dirty = os.environ.get("MAXSIM_BENCH_GIT_DIRTY", "false").lower() in {
        "1", "true", "yes",
    }

    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name()
        sm_major, sm_minor = torch.cuda.get_device_capability()
        compute_capability = f"{sm_major}.{sm_minor}"
    else:
        import platform

        gpu_name = f"Apple Silicon MPS ({platform.machine()})"
        compute_capability = "mps"

    print(f"[cuda_bench_matrix] device  = {DEVICE}")
    print(f"[cuda_bench_matrix] gpu     = {gpu_name}  cc={compute_capability}")
    print(f"[cuda_bench_matrix] torch   = {torch.__version__}  cuda={torch.version.cuda}")
    print(f"[cuda_bench_matrix] iters   = {n_iter}  warmup = {n_warmup}  repeats = {n_repeats}")
    print(f"[cuda_bench_matrix] commit  = {git_commit}{' (dirty)' if git_dirty else ''}")

    if DEVICE == "cuda":
        # CUDA JIT-compiles maxsim.cu via torch.utils.cpp_extension, which needs
        # ninja. Metal loads the prebuilt build/ tree instead, so it has no such
        # dependency — keep the install out of the Metal path.
        _ensure("ninja")
        _setup_bridge(repo)
    else:
        _setup_metal_import(repo)
    import maxsim

    dtypes = [torch.float16, torch.bfloat16]
    contrastive = [
        ("Contrastive", dict(Nq=32, Nb=32, Lq=32, Ld=80, dim=128)),
        ("LongDocs", dict(Nq=32, Nb=32, Lq=32, Ld=512, dim=128)),
        ("BigBatch", dict(Nq=64, Nb=64, Lq=32, Ld=128, dim=128)),
    ]
    padded = [
        ("Rerank", dict(B=32, C=50, Lq=32, Ld=180, dim=128)),
        ("HeavyRerank", dict(B=32, C=100, Lq=32, Ld=256, dim=128)),
    ]
    packed = [
        ("PackedRerank", dict(B=32, C=50, Lq=32, Ld=180, dim=128)),
        ("PackedHeavyRerank", dict(B=32, C=100, Lq=32, Ld=256, dim=128)),
    ]

    def _one_matrix() -> list[Result]:
        rs: list[Result] = []
        for dtype in dtypes:
            for name, shape in contrastive:
                rs.append(_bench_contrastive(maxsim, name, shape, dtype, n_iter, n_warmup))
            for name, shape in padded:
                rs.append(_bench_padded_infer(maxsim, name, shape, dtype, n_iter, n_warmup))
            for name, shape in packed:
                rs.append(_bench_packed_infer(maxsim, name, shape, dtype, n_iter, n_warmup))
        return rs

    started_at = dt.datetime.now(dt.timezone.utc)
    t0 = time.perf_counter()

    all_results: list[list[Result]] = []
    for rep in range(n_repeats):
        print(f"[cuda_bench_matrix] run {rep + 1}/{n_repeats} starting")
        all_results.append(_one_matrix())
        print(f"[cuda_bench_matrix] run {rep + 1}/{n_repeats} done")

    if n_repeats == 1:
        final = all_results[0]
        variance = [{"n_repeats": 1}] * len(final)
    else:
        final, variance = _summarise_repeats(all_results)

    wall = time.perf_counter() - t0
    print(f"[cuda_bench_matrix] total wall: {wall:.1f}s")

    # JSON output. Filename derived from the GPU name slug; rerunning on the
    # same hardware overwrites, with metadata (commit, started_at) describing
    # what the current artifact represents.
    out_dir = repo / "bench_results" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_slugify_gpu(gpu_name)}.json"

    results_with_variance = [
        {**r.to_dict(), "variance": v}
        for r, v in zip(final, variance)
    ]

    payload = {
        "schema_version": 1,
        "kernel": {
            "repo": "erikkaum/maxsim",
            "version": 2,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
        "machine": {
            "gpu_name": gpu_name,
            "compute_capability": compute_capability,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "bench": {
            "iters_per_repeat": n_iter,
            "warmup": n_warmup,
            "repeats": n_repeats,
            "started_at": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_seconds": round(wall, 1),
        },
        "results": results_with_variance,
    }

    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[cuda_bench_matrix] wrote {out_path}")

    return 0 if all(r.correct for r in final) else 1


if __name__ == "__main__":
    sys.exit(main())
