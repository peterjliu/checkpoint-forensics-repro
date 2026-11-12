"""Known-answer and layout tests for the Philox / PyTorch-CUDA normal_ emulator."""

import numpy as np

from forensics.philox import normal_at, philox4x32_10


def _block(ctr: tuple[int, int, int, int], key: tuple[int, int]) -> list[str]:
    c = [np.array([v], dtype=np.uint32) for v in ctr]
    out = philox4x32_10(c[0], c[1], c[2], c[3], key[0], key[1])
    return [f"{int(w[0]):08x}" for w in out]


def test_philox_known_answers_from_random123() -> None:
    assert ["6627e8d5", "e169c58d", "bc57ac4c", "9b00dbd8"] == _block((0, 0, 0, 0), (0, 0))
    f = 0xFFFFFFFF
    assert ["408f276d", "41c83b0e", "a20bc7c6", "6d5451fd"] == _block((f, f, f, f), (f, f))
    assert ["d16cfe09", "94fdcceb", "5001e420", "24126ea1"] == _block(
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0)
    )


def test_thread_sequence_is_layout_independent_but_positions_move() -> None:
    # Thread 3's k-th value lands at 3 + k*T. Same values, different positions.
    seed = 7
    ks = np.arange(8, dtype=np.int64)
    a = normal_at(seed, 48, 3 + ks * 48)
    b = normal_at(seed, 40, 3 + ks * 40)
    assert a.tolist() == b.tolist()
    # ...and the same positions under a different thread count disagree.
    pos = np.arange(1000, 2000, dtype=np.int64)
    assert not np.allclose(normal_at(seed, 48, pos), normal_at(seed, 40, pos))


def test_normals_are_standard_normal() -> None:
    x = normal_at(12345, 270336, np.arange(2_000_000, dtype=np.int64))
    assert abs(float(x.mean())) < 0.005
    assert abs(float(x.std()) - 1.0) < 0.005


def test_offset_shifts_the_per_thread_stream_by_whole_calls() -> None:
    # One curand_normal4 call yields 4 values; offset is counted in calls.
    t = 16
    pos = np.arange(0, 64, dtype=np.int64)
    base = normal_at(1, t, np.arange(0, 128, dtype=np.int64))
    shifted = normal_at(1, t, pos, offset_calls=1)
    assert base[4 * t : 4 * t + 64].tolist() == shifted.tolist()


def test_trunc_normal_is_cut_at_three_std_and_shares_the_layout() -> None:
    from forensics.philox import values_at

    pos = np.arange(1_000_000, dtype=np.int64)
    x = values_at(11, 270336, pos, dist="trunc_normal3")
    assert float(np.abs(x).max()) <= 3.0
    assert abs(float(x.std()) - 0.9866) < 0.003  # std of a normal truncated at +-3
    # Same placement rule as normal_: thread 3's k-th value sits at 3 + k*T.
    ks = np.arange(8, dtype=np.int64)
    a = values_at(11, 48, 3 + ks * 48, dist="trunc_normal3")
    b = values_at(11, 40, 3 + ks * 40, dist="trunc_normal3")
    assert a.tolist() == b.tolist()
