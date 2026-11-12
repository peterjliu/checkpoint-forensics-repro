"""Score candidate GPU layouts (SM counts) for a known seed against published weights."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from forensics.philox import F32, I64, Dist, threads_for_sms, values_at

PRESERVED_R = 0.99


@dataclass(frozen=True)
class LayoutScore:
    sms: int
    pooled_r: float  # correlation over every compared value
    sigma: float  # pooled_r * sqrt(N): distance from chance in standard deviations
    n_values: int
    preserved_rows: I64  # row ids whose correlation exceeds the untrained threshold
    row_r: NDArray[np.float64]


def row_positions(row_ids: I64, n_cols_total: int, n_cols_used: int) -> I64:
    """Flat row-major positions of the first ``n_cols_used`` columns of each row."""
    return row_ids[:, None] * n_cols_total + np.arange(n_cols_used, dtype=np.int64)[None, :]


def _standardize(x: NDArray[np.float32]) -> NDArray[np.float64]:
    y = x.astype(np.float64)
    y -= y.mean(axis=1, keepdims=True)
    norm = np.sqrt((y * y).sum(axis=1, keepdims=True))
    norm[norm == 0] = 1.0
    return y / norm


def scan_layouts(
    weights: F32,
    row_ids: I64,
    n_cols_total: int,
    seed: int,
    sms_candidates: list[int],
    offset_calls: int = 0,
    dist: Dist = "normal",
    preserved_r: float = PRESERVED_R,
) -> list[LayoutScore]:
    """Correlate ``weights[i, :]`` (the first columns of row ``row_ids[i]``) with each layout's values."""
    # The first pass of every thread (positions below the thread count) is the
    # same on every layout, so those rows cannot tell layouts apart: drop them.
    shared = max(threads_for_sms(s) for s in sms_candidates)
    discriminating = row_ids * n_cols_total >= shared
    weights, row_ids = weights[discriminating], row_ids[discriminating]
    n_rows, n_cols = weights.shape
    pos = row_positions(row_ids, n_cols_total, n_cols)
    w = _standardize(weights)
    out: list[LayoutScore] = []
    for sms in sms_candidates:
        gen = values_at(seed, threads_for_sms(sms), pos.reshape(-1), offset_calls, dist).reshape(n_rows, n_cols)
        row_r: NDArray[np.float64] = (w * _standardize(gen)).sum(axis=1)
        n = n_rows * n_cols
        pooled = float(row_r.mean())  # rows are unit-norm, so the mean row r is the pooled r
        out.append(LayoutScore(sms, pooled, pooled * float(np.sqrt(n)), n, row_ids[row_r > preserved_r], row_r))
    return out


def to_bf16_bits(x: F32) -> NDArray[np.uint16]:
    """Round float32 to bfloat16 (nearest, ties to even) and return the 16-bit patterns."""
    bits = x.view(np.uint32).astype(np.uint64)
    rounded = (bits + np.uint64(0x7FFF) + ((bits >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)
    return rounded.astype(np.uint16)


def find_std(published: F32, generated: F32) -> float:
    """Least-squares scale s minimising |published - s * generated|."""
    g = generated.astype(np.float64).reshape(-1)
    p = published.astype(np.float64).reshape(-1)
    return float((g * p).sum() / (g * g).sum())


def load_embedding(bin_path: Path) -> tuple[NDArray[np.uint16], tuple[int, int]]:
    meta = json.loads(bin_path.with_suffix(".json").read_text())
    if meta["dtype"] != "BF16":
        raise ValueError(f"expected BF16, got {meta['dtype']}")
    rows, cols = int(meta["shape"][0]), int(meta["shape"][1])
    return np.fromfile(bin_path, dtype=np.uint16).reshape(rows, cols), (rows, cols)


def bf16_to_f32(bits: NDArray[np.uint16]) -> F32:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("embed", type=Path)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sms", type=int, nargs="+", default=[108, 132, 148, 152])
    ap.add_argument("--sms-range", type=int, nargs=2, metavar=("LO", "HI"), help="scan every SM count in [LO, HI]")
    ap.add_argument("--cols", type=int, default=128, help="columns per row used in the all-rows scan")
    ap.add_argument("--offset-calls", type=int, default=0)
    ap.add_argument("--dist", choices=["normal", "trunc_normal3"], default="normal")
    ap.add_argument("--preserved-r", type=float, default=PRESERVED_R, help="row correlation that counts as untrained")
    ap.add_argument("--std", type=float, default=0.0, help="init std for the bit-exact check (0 = estimate)")
    args = ap.parse_args()

    if args.sms_range:
        args.sms = list(range(args.sms_range[0], args.sms_range[1] + 1))
    bits, (rows, cols) = load_embedding(args.embed)
    row_ids = np.arange(rows, dtype=np.int64)
    head = bf16_to_f32(np.ascontiguousarray(bits[:, : args.cols]))
    print(f"{args.embed}: {rows} x {cols}; scanning all rows x first {args.cols} cols, seed {args.seed}")
    results = scan_layouts(head, row_ids, cols, args.seed, args.sms, args.offset_calls, args.dist, args.preserved_r)
    for r in sorted(results, key=lambda r: -abs(r.sigma))[:8]:
        print(f"  SMs {r.sms:4d}: pooled r {r.pooled_r:+.5f}  sigma {r.sigma:9.1f}  rows with r>{args.preserved_r}: {len(r.preserved_rows)}")

    best = max(results, key=lambda r: r.sigma)
    if len(best.preserved_rows) == 0:
        print("no preserved rows found")
        return
    # Full-width check on the preserved rows, at every candidate layout.
    keep = best.preserved_rows
    full = bf16_to_f32(np.ascontiguousarray(bits[keep, :]))
    print(f"full-width check on {len(keep)} preserved rows ({full.size:,d} values):")
    full_scores = scan_layouts(full, keep, cols, args.seed, args.sms, args.offset_calls, args.dist, args.preserved_r)
    for r in sorted(full_scores, key=lambda r: -abs(r.sigma))[:8]:
        print(f"  SMs {r.sms:4d}: pooled r {r.pooled_r:+.6f}  sigma {r.sigma:9.1f}  rows r>{args.preserved_r}: {len(r.preserved_rows)}")
    gen = values_at(args.seed, threads_for_sms(best.sms), row_positions(keep, cols, cols).reshape(-1), args.offset_calls, args.dist)
    gen = gen.reshape(full.shape)
    std = args.std or find_std(full, gen)
    same = to_bf16_bits(gen * np.float32(std)) == np.ascontiguousarray(bits[keep, :])
    print(f"best layout {best.sms} SMs; fitted std {find_std(full, gen):.6f}, using {std:.6f}")
    print(f"  bit-identical values: {int(same.sum()):,d} / {same.size:,d}; rows fully bit-identical: {int(same.all(axis=1).sum())} / {len(keep)}")
    print(f"  preserved row ids: {keep.tolist()[:40]}{' ...' if len(keep) > 40 else ''}")


if __name__ == "__main__":
    main()
