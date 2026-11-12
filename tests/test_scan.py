"""The layout scan must pick out the true thread count and the preserved rows."""

import numpy as np

from forensics.philox import normal_at, threads_for_sms
from forensics.scan import find_std, row_positions, scan_layouts, to_bf16_bits


def _fake_embedding(seed: int, sms: int, rows: int, cols: int, preserved: list[int]) -> np.ndarray:
    pos = np.arange(rows * cols, dtype=np.int64)
    w = (normal_at(seed, threads_for_sms(sms), pos) * np.float32(0.02)).reshape(rows, cols)
    noise = np.random.default_rng(0).normal(0, 0.05, size=w.shape).astype(np.float32)
    trained = np.ones(rows, dtype=bool)
    trained[preserved] = False
    w[trained] += noise[trained]
    return w


def test_scan_finds_true_layout_and_preserved_rows() -> None:
    rows, cols = 3000, 512  # rows below ~608 sit in the first thread pass, identical on every layout
    w = _fake_embedding(seed=99, sms=148, rows=rows, cols=cols, preserved=[700, 1700, 2999])
    row_ids = np.arange(rows, dtype=np.int64)
    results = scan_layouts(w, row_ids, cols, seed=99, sms_candidates=[108, 132, 148, 152])
    by_sms = {r.sms: r for r in results}
    assert [700, 1700, 2999] == by_sms[148].preserved_rows.tolist()
    assert [0, 0, 0] == [len(by_sms[s].preserved_rows) for s in (108, 132, 152)]
    assert by_sms[148].sigma > 100
    assert all(abs(by_sms[s].sigma) < 6 for s in (108, 132, 152))


def test_wrong_seed_scores_nothing() -> None:
    rows, cols = 3000, 512
    w = _fake_embedding(seed=99, sms=148, rows=rows, cols=cols, preserved=[5])
    results = scan_layouts(w, np.arange(rows, dtype=np.int64), cols, seed=100, sms_candidates=[148])
    assert 0 == len(results[0].preserved_rows)
    assert abs(results[0].sigma) < 6


def test_row_positions_are_row_major() -> None:
    assert [[20, 21, 22], [50, 51, 52]] == row_positions(np.array([2, 5]), 10, 3).tolist()


def test_bf16_rounding_is_nearest_even() -> None:
    x = np.array([1.0, 1.0 + 2**-8, 1.0 + 3 * 2**-8, -0.02], dtype=np.float32)
    # 1+2^-8 is a tie -> rounds to even (down); 1+3*2^-8 is a tie -> rounds up to even.
    assert [0x3F80, 0x3F80, 0x3F82, 0xBCA4] == to_bf16_bits(x).tolist()


def test_find_std_recovers_scale() -> None:
    g = normal_at(1, 1024, np.arange(4096, dtype=np.int64))
    assert abs(find_std(g * np.float32(0.02), g) - 0.02) < 1e-6
