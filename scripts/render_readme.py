# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Render bench result tables into the README from `bench_results/v2/*.json`.

Source of truth: the JSON artifacts emitted by `cuda_bench_matrix.py`. This
script reads them and rewrites the README sections between marker comments:

    <!-- BENCH:cross-gpu-contrastive -->
        (rendered cross-GPU Contrastive fp16 table)
    <!-- /BENCH -->

    <!-- BENCH:full-matrix-h200 -->
        (rendered full H200 matrix)
    <!-- /BENCH -->

    <!-- BENCH:full-matrix-a100 -->
        (rendered full A100 matrix)
    <!-- /BENCH -->

Run with::

    just bench-render
    # or: python scripts/render_readme.py

Idempotent — running twice with no JSON change produces no diff. Exits 1 if
any expected marker is missing or any expected JSON file is missing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO / "bench_results" / "v2"
README = REPO / "README.md"

# GPUs we expect to find in the cross-GPU table, in display order. Slug must
# match what `cuda_bench_matrix._slugify_gpu` produces for that GPU's
# `gpu_name` (vendor prefix stripped, lowercased, non-alnum → "-").
CROSS_GPU_ORDER = [
    ("h200", "H200"),
    ("a100-sxm4-80gb", "A100 SXM4 80GB"),
    ("l40s", "L40S"),
    ("l4", "L4"),
    ("a10g", "A10G"),
    ("apple-silicon-mps-arm64", "Apple Silicon MPS"),
]

# Which GPUs get a rendered full benchmark table, keyed by marker name.
FULL_MATRIX_GPUS = {
    "full-matrix-h200": "h200",
    "full-matrix-a100": "a100-sxm4-80gb",
}
SURFACE_ORDER = {
    "contrastive_train": 0,
    "padded_infer": 1,
    "packed_infer": 2,
}
PRESET_ORDER = {
    "Contrastive": 0,
    "LongDocs": 1,
    "BigBatch": 2,
    "Rerank": 3,
    "HeavyRerank": 4,
    "PackedRerank": 5,
    "PackedHeavyRerank": 6,
}
DTYPE_ORDER = {"fp16": 0, "bf16": 1}


def _fmt_ms(v) -> str:
    return "—" if v is None else f"{v:.3f} ms"


def _fmt_x(v) -> str:
    return "—" if v is None else f"{v:.2f}×"


def _fmt_retained(v) -> str:
    return "—" if v is None else f"1/{v:.0f}"


def _fmt_shape(shape: dict) -> str:
    """Render a structured shape dict as `key=value, ...` for the README."""
    return ", ".join(f"{k}={v}" for k, v in shape.items())


def _load_json(slug: str) -> dict | None:
    path = BENCH_DIR / f"{slug}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _result_matching(payload: dict, *, surface: str, preset: str, dtype: str) -> dict | None:
    results = [r for r in payload["results"] if r["surface"] in SURFACE_ORDER]
    results.sort(key=lambda r: (
        SURFACE_ORDER[r["surface"]],
        PRESET_ORDER.get(r["preset"], 99),
        DTYPE_ORDER.get(r["dtype"], 99),
    ))
    for r in results:
        if r["surface"] == surface and r["preset"] == preset and r["dtype"] == dtype:
            return r
    return None


def _render_cross_gpu_contrastive() -> str:
    """Cross-GPU table: one row per GPU we have JSON for. Contrastive / fp16."""
    rows: list[str] = []
    rows.append(
        "| GPU | sm | maxsim step | naive step | speedup |"
    )
    rows.append("| --- | --- | ---: | ---: | ---: |")

    for slug, label in CROSS_GPU_ORDER:
        payload = _load_json(slug)
        if payload is None:
            continue
        result = _result_matching(
            payload,
            surface="contrastive_train",
            preset="Contrastive",
            dtype="fp16",
        )
        if result is None:
            continue
        sm = payload["machine"]["compute_capability"]
        # H200 (and any future PTX-JIT class) gets a parenthetical so readers
        # know there's no native sm_X build yet. Heuristic: anything sm ≥ 9.0
        # is JIT today.
        try:
            sm_major = int(sm.split(".")[0])
            label_display = f"{label} (PTX-JIT)" if sm_major >= 9 else label
        except ValueError:
            label_display = label
        rows.append(
            f"| {label_display} | sm_{sm.replace('.', '')} | "
            f"{_fmt_ms(result['maxsim_step_ms'])} | "
            f"{_fmt_ms(result['naive_step_ms'])} | "
            f"{_fmt_x(result['step_speedup'])} |"
        )
    return "\n".join(rows)


def _render_full_matrix(slug: str) -> str:
    payload = _load_json(slug)
    if payload is None:
        raise FileNotFoundError(
            f"missing {BENCH_DIR / (slug + '.json')} for the full-matrix table"
        )
    rows: list[str] = []
    rows.append(
        "| Surface | Preset | Shape | dtype | maxsim | PyTorch | speedup | "
        "padded | bwd× | peak× | retained state |"
    )
    rows.append(
        "| --- | --- | --- | --- | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |"
    )
    results = [r for r in payload["results"] if r["surface"] in SURFACE_ORDER]
    results.sort(key=lambda r: (
        SURFACE_ORDER[r["surface"]],
        PRESET_ORDER.get(r["preset"], 99),
        DTYPE_ORDER.get(r["dtype"], 99),
    ))
    for r in results:
        rows.append(
            f"| {r['surface']} | {r['preset']} | `{_fmt_shape(r['shape'])}` | "
            f"{r['dtype']} | {_fmt_ms(r['maxsim_step_ms'])} | "
            f"{_fmt_ms(r['naive_step_ms'])} | "
            f"{_fmt_x(r['step_speedup'])} | "
            f"{_fmt_ms(r.get('padded_ms'))} | "
            f"{_fmt_x(r['bwd_speedup'])} | "
            f"{_fmt_x(r['peak_ratio'])} | "
            f"{_fmt_retained(r['retained_reduction'])} |"
        )
    return "\n".join(rows)


def _replace_marker(text: str, marker: str, body: str) -> tuple[str, bool]:
    """Replace the contents between
        <!-- BENCH:{marker} -->
        ...
        <!-- /BENCH -->
    with `body`. Returns (new_text, replaced)."""
    pattern = re.compile(
        rf"(<!--\s*BENCH:{re.escape(marker)}\s*-->\n)"
        rf"(.*?)"
        rf"(\n<!--\s*/BENCH\s*-->)",
        re.DOTALL,
    )
    new_text, n = pattern.subn(
        lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}",
        text,
        count=1,
    )
    return new_text, bool(n)


def main() -> int:
    if not README.exists():
        print(f"ERROR: {README} not found", file=sys.stderr)
        return 1

    src = README.read_text()
    renders = [
        ("cross-gpu-contrastive", _render_cross_gpu_contrastive()),
        *[
            (marker, _render_full_matrix(slug))
            for marker, slug in FULL_MATRIX_GPUS.items()
        ],
    ]

    for marker, body in renders:
        src, replaced = _replace_marker(src, marker, body)
        if not replaced:
            print(
                f"ERROR: marker `BENCH:{marker}` not found in README. "
                f"Add `<!-- BENCH:{marker} -->` and `<!-- /BENCH -->` "
                f"around the block to render.",
                file=sys.stderr,
            )
            return 1
        print(f"[render_readme] BENCH:{marker} rendered")

    if README.read_text() == src:
        print("[render_readme] README unchanged")
        return 0

    README.write_text(src)
    print(f"[render_readme] wrote {README}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
