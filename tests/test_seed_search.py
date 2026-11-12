"""Blind recovery of seed, layout and generator offset from weights alone."""

import numpy as np

from forensics.philox import normal_at, philox4x32_10, threads_for_sms
from forensics.seed_search import Hit, make_probe, search_offsets, search_seeds

ROWS, COLS = 1500, 1024
UNTRAINED = [900, 901, 977, 1400]


def _z(seed: int, sms: int, offset_calls: int = 0) -> np.ndarray:
    pos = np.arange(ROWS * COLS, dtype=np.int64)
    return normal_at(seed, threads_for_sms(sms), pos, offset_calls).reshape(ROWS, COLS)


def test_philox_accepts_per_element_keys() -> None:
    z = np.zeros(2, dtype=np.uint32)
    keys = np.array([0, 0xA4093822], dtype=np.uint32)
    out = philox4x32_10(z, z, z, z, keys, np.zeros(2, dtype=np.uint32))
    assert "6627e8d5" == f"{int(out[0][0]):08x}"
    single = philox4x32_10(z[:1], z[:1], z[:1], z[:1], 0xA4093822, 0)
    assert int(single[0][0]) == int(out[0][1])


def test_search_recovers_seed_and_layout() -> None:
    z = _z(seed=70_123, sms=148)
    probes = [make_probe(z, sms, UNTRAINED) for sms in (132, 148, 152)]
    assert [Hit(70_123, 148, 0)] == search_seeds(probes, lo=0, hi=200_000, chunk=50_000, workers=2)


def test_search_finds_nothing_in_unrelated_values() -> None:
    z = np.random.default_rng(3).normal(size=(ROWS, COLS)).astype(np.float32)
    probes = [make_probe(z, sms, UNTRAINED) for sms in (132, 148)]
    assert [] == search_seeds(probes, lo=0, hi=200_000, chunk=50_000, workers=2)


def test_offset_search_recovers_generator_offset_for_known_seed() -> None:
    z = _z(seed=5, sms=152, offset_calls=31_337)
    probes = [make_probe(z, sms, UNTRAINED) for sms in (148, 152)]
    assert [Hit(5, 152, 31_337)] == search_offsets(probes, seed=5, lo=0, hi=100_000, chunk=25_000, workers=2)


def test_search_recovers_seed_for_truncated_normal_init() -> None:
    from forensics.philox import values_at

    pos = np.arange(ROWS * COLS, dtype=np.int64)
    z = values_at(4242, threads_for_sms(152), pos, dist="trunc_normal3").reshape(ROWS, COLS)
    probes = [make_probe(z, sms, UNTRAINED, dist="trunc_normal3") for sms in (148, 152)]
    assert [Hit(4242, 152, 0)] == search_seeds(probes, lo=0, hi=50_000, chunk=25_000, workers=2)
    normal_probes = [make_probe(z, sms, UNTRAINED) for sms in (148, 152)]
    assert [] == search_seeds(normal_probes, lo=0, hi=50_000, chunk=25_000, workers=2)
