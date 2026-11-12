"""Blind search for the (seed, GPU layout, generator offset) behind untrained weights.

A probe is a handful of published values at known flat positions. Under a
candidate layout each position maps to one (thread, call, component) of the
Philox stream, so a candidate seed/offset either reproduces every value or not.
Candidates are filtered one point at a time, so only the first point costs a
full pass.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from forensics.philox import F32, I64, Dist, block_values, philox4x32_10, threads_for_sms, values_at
from forensics.scan import bf16_to_f32, load_embedding, row_positions

U64 = NDArray[np.uint64]


@dataclass(frozen=True)
class Point:
    thread: int
    call: int  # curand_normal4 call index within the thread, before any generator offset
    comp: int  # which of the call's 4 normals
    target: float  # published value in units of the init std


@dataclass(frozen=True)
class Probe:
    sms: int
    points: tuple[Point, ...]
    dist: Dist = "normal"


@dataclass(frozen=True)
class Tolerance:
    rel: float = 0.006
    abs: float = 0.001


@dataclass(frozen=True)
class Hit:
    seed: int
    sms: int
    offset_calls: int


def make_probe(z: F32, sms: int, rows: list[int], n_points: int = 8, dist: Dist = "normal") -> Probe:
    """Probe from the leading columns of ``rows`` of ``z`` (weights / init std)."""
    n_cols = z.shape[1]
    threads = threads_for_sms(sms)
    points: list[Point] = []
    for i in range(n_points):
        row, col = rows[i % len(rows)], i // len(rows)
        p = row * n_cols + col
        k = p // threads
        points.append(Point(p % threads, k // 4, k % 4, float(z[row, col])))
    return Probe(sms, tuple(points), dist)


def _values(pt: Point, seeds: U64, offsets: U64, dist: Dist) -> F32:
    call = offsets + np.uint64(pt.call)
    w = philox4x32_10(
        (call & np.uint64(0xFFFFFFFF)).astype(np.uint32),
        (call >> np.uint64(32)).astype(np.uint32),
        np.full(1, pt.thread, dtype=np.uint32),
        np.zeros(1, dtype=np.uint32),
        (seeds & np.uint64(0xFFFFFFFF)).astype(np.uint32),
        (seeds >> np.uint64(32)).astype(np.uint32),
    )
    return block_values(w, dist)[pt.comp]


def matching(probe: Probe, seeds: U64, offsets: U64, tol: Tolerance) -> I64:
    """Indices i where (seeds[i], offsets[i]) reproduces every point of the probe."""
    alive = np.arange(len(seeds), dtype=np.int64)
    for pt in probe.points:
        gen = _values(pt, seeds[alive], offsets[alive], probe.dist)
        ok = np.abs(gen - np.float32(pt.target)) <= tol.abs + tol.rel * abs(pt.target)
        alive = alive[ok]
        if len(alive) == 0:
            break
    return alive


_SLICE = 1 << 16  # candidates per vectorised step: small enough to stay in cache


def _seed_chunk(args: tuple[list[Probe], int, int, int, Tolerance]) -> list[Hit]:
    probes, lo, hi, offset_calls, tol = args
    hits: list[Hit] = []
    for a in range(lo, hi, _SLICE):
        seeds = np.arange(a, min(a + _SLICE, hi), dtype=np.uint64)
        offsets = np.full(len(seeds), offset_calls, dtype=np.uint64)
        hits.extend(Hit(int(seeds[i]), pr.sms, offset_calls) for pr in probes for i in matching(pr, seeds, offsets, tol))
    return hits


def _offset_chunk(args: tuple[list[Probe], int, int, int, Tolerance]) -> list[Hit]:
    probes, lo, hi, seed, tol = args
    hits: list[Hit] = []
    for a in range(lo, hi, _SLICE):
        offsets = np.arange(a, min(a + _SLICE, hi), dtype=np.uint64)
        seeds = np.full(len(offsets), seed, dtype=np.uint64)
        hits.extend(Hit(seed, pr.sms, int(offsets[i])) for pr in probes for i in matching(pr, seeds, offsets, tol))
    return hits


def search_seeds(
    probes: list[Probe],
    lo: int = 0,
    hi: int = 1 << 32,
    offset_calls: int = 0,
    tol: Tolerance = Tolerance(),
    chunk: int = 1 << 21,
    workers: int = 16,
) -> list[Hit]:
    """Every seed in [lo, hi) that reproduces a probe at a fixed generator offset."""
    jobs = [(probes, a, min(a + chunk, hi), offset_calls, tol) for a in range(lo, hi, chunk)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [h for part in pool.map(_seed_chunk, jobs, chunksize=4) for h in part]


def search_offsets(
    probes: list[Probe],
    seed: int,
    lo: int = 0,
    hi: int = 1 << 24,
    tol: Tolerance = Tolerance(),
    chunk: int = 1 << 21,
    workers: int = 16,
) -> list[Hit]:
    """Every generator offset in [lo, hi) that reproduces a probe for a known seed."""
    jobs = [(probes, a, min(a + chunk, hi), seed, tol) for a in range(lo, hi, chunk)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [h for part in pool.map(_offset_chunk, jobs, chunksize=1) for h in part]


def verify(bits: NDArray[np.uint16], hit: Hit, rows: list[int], dist: Dist = "normal") -> float:
    """Correlation between the full published ``rows`` and the hit's regenerated values."""
    n_cols = bits.shape[1]
    ids = np.array(rows, dtype=np.int64)
    pub = bf16_to_f32(np.ascontiguousarray(bits[ids])).astype(np.float64).reshape(-1)
    gen = values_at(hit.seed, threads_for_sms(hit.sms), row_positions(ids, n_cols, n_cols).reshape(-1), hit.offset_calls, dist)
    return float(np.corrcoef(pub, gen.astype(np.float64))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("embed", type=Path)
    ap.add_argument("--rows", type=int, nargs="+", required=True, help="rows believed untrained")
    ap.add_argument("--std", type=float, default=0.0, help="init std (0 = use each row's RMS)")
    ap.add_argument("--sms", type=int, nargs="+", default=[108, 132, 148, 152])
    ap.add_argument("--sms-range", type=int, nargs=2, metavar=("LO", "HI"))
    ap.add_argument("--seed", type=int, help="known seed: search generator offsets instead of seeds")
    ap.add_argument("--offset-calls", type=int, default=0)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=1 << 32)
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--dist", choices=["normal", "trunc_normal3"], default="normal")
    ap.add_argument("--tol-rel", type=float, default=0.006)
    ap.add_argument("--tol-abs", type=float, default=0.001)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    if args.sms_range:
        args.sms = list(range(args.sms_range[0], args.sms_range[1] + 1))

    bits, (_, cols) = load_embedding(args.embed)
    # Sparse copy holding just the probed rows, scaled to unit std.
    z = np.zeros((max(args.rows) + 1, cols), dtype=np.float32)
    for r in args.rows:
        row = bf16_to_f32(np.ascontiguousarray(bits[r])).astype(np.float64)
        std = args.std or float(np.sqrt((row * row).mean()))
        z[r] = (row / std).astype(np.float32)
    probes = [make_probe(z, sms, args.rows, args.points, args.dist) for sms in args.sms]
    tol = Tolerance(args.tol_rel, args.tol_abs)
    if args.seed is None:
        print(f"searching seeds [{args.lo}, {args.hi}) x {len(probes)} layouts, offset {args.offset_calls}")
        hits = search_seeds(probes, args.lo, args.hi, args.offset_calls, tol, workers=args.workers)
    else:
        print(f"seed {args.seed}: searching offsets [{args.lo}, {args.hi}) x {len(probes)} layouts")
        hits = search_offsets(probes, args.seed, args.lo, args.hi, tol, workers=args.workers)
    for h in hits:
        print(f"HIT seed {h.seed} sms {h.sms} offset_calls {h.offset_calls}: r over probed rows = {verify(bits, h, args.rows, args.dist):+.5f}")
    print(f"search complete, {len(hits)} hits")


if __name__ == "__main__":
    main()
