"""CPU emulation of PyTorch's CUDA ``normal_`` fill (cuRAND Philox4x32-10).

PyTorch launches ``threads = min(SMs * (2048 // 256), ceil(numel / 256)) * 256``
GPU threads (aten/src/ATen/native/cuda/DistributionTemplates.h). Thread ``i``
seeds Philox with ``curand_init(seed, subsequence=i, offset)`` and writes the
k-th normal it draws to flat position ``i + k * threads``. Each
``curand_normal4`` call turns one Philox block (4 words) into 4 normals by two
Box-Muller transforms.

So the value at flat position ``p`` comes from:
    thread = p % threads, k = p // threads, call = k // 4, component = k % 4
    block  = philox(ctr=(offset_calls + call, 0, thread, 0), key=(seed, 0))
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr, ndtri

U32 = NDArray[np.uint32]
F32 = NDArray[np.float32]
I64 = NDArray[np.int64]

# How the fill turned Philox words into weights (in units of the init std):
#   normal        - Tensor.normal_ (Box-Muller)
#   trunc_normal3 - nn.init.trunc_normal_ cut at +-3 std (uniform_ then erfinv_)
Dist = Literal["normal", "trunc_normal3"]
TRUNC_BOUND = 3.0

_M0 = np.uint64(0xD2511F53)
_M1 = np.uint64(0xCD9E8D57)
_W0 = np.uint64(0x9E3779B9)
_W1 = np.uint64(0xBB67AE85)
_MASK = np.uint64(0xFFFFFFFF)
_SHIFT = np.uint64(32)

THREADS_PER_SM = 2048
_TWO_POW32_INV = np.float32(2.3283064e-10)
_TWO_POW32_INV_2PI = np.float32(2.3283064e-10 * 6.2831855)


def threads_for_sms(sms: int) -> int:
    """Thread count PyTorch uses for a large fill on a GPU with ``sms`` SMs."""
    return sms * THREADS_PER_SM


def philox4x32_10(c0: U32, c1: U32, c2: U32, c3: U32, key0: int | U32, key1: int | U32) -> tuple[U32, U32, U32, U32]:
    """Ten-round Philox4x32. Counters and keys broadcast: either may be scalar or array."""
    x0 = np.asarray(c0).astype(np.uint64)
    x1 = np.asarray(c1).astype(np.uint64)
    x2 = np.asarray(c2).astype(np.uint64)
    x3 = np.asarray(c3).astype(np.uint64)
    k0 = np.asarray(key0).astype(np.uint64) & _MASK
    k1 = np.asarray(key1).astype(np.uint64) & _MASK
    for _ in range(10):
        p0 = _M0 * x0
        p1 = _M1 * x2
        n0 = (p1 >> _SHIFT) ^ x1 ^ k0
        n1 = p1 & _MASK
        n2 = (p0 >> _SHIFT) ^ x3 ^ k1
        n3 = p0 & _MASK
        x0, x1, x2, x3 = n0, n1, n2, n3
        k0 = (k0 + _W0) & _MASK
        k1 = (k1 + _W1) & _MASK
    return (x0.astype(np.uint32), x1.astype(np.uint32), x2.astype(np.uint32), x3.astype(np.uint32))


def box_muller(x: U32, y: U32) -> tuple[F32, F32]:
    """cuRAND ``_curand_box_muller``: returns (s*sin(v), s*cos(v)) in float32."""
    half_u = np.float32(_TWO_POW32_INV / np.float32(2))
    half_v = np.float32(_TWO_POW32_INV_2PI / np.float32(2))
    u: F32 = x.astype(np.float32) * _TWO_POW32_INV + half_u
    v: F32 = y.astype(np.float32) * _TWO_POW32_INV_2PI + half_v
    s: F32 = np.sqrt(np.float32(-2) * np.log(u)).astype(np.float32)
    return (np.sin(v) * s).astype(np.float32), (np.cos(v) * s).astype(np.float32)


def uniform(x: U32) -> F32:
    """cuRAND ``curand_uniform``: a 32-bit word mapped into (0, 1]."""
    return (x.astype(np.float32) * _TWO_POW32_INV + np.float32(_TWO_POW32_INV / np.float32(2))).astype(np.float32)


def trunc_normal_from_uniform(u: F32) -> F32:
    """``trunc_normal_``: uniform_(2l-1, 2u-1) -> erfinv_ -> * sqrt(2), in std units.

    sqrt(2) * erfinv(2p - 1) is the normal quantile of p, so this is ndtri of a
    uniform spread over [cdf(-bound), cdf(bound)].
    """
    lo = float(ndtr(-TRUNC_BOUND))
    hi = float(ndtr(TRUNC_BOUND))
    p = lo + u.astype(np.float64) * (hi - lo)
    return np.clip(ndtri(p), -TRUNC_BOUND, TRUNC_BOUND).astype(np.float32)


def block_values(w: tuple[U32, U32, U32, U32], dist: Dist) -> tuple[F32, F32, F32, F32]:
    """The four weights (std units) one Philox block produces under ``dist``."""
    if dist == "normal":
        a_sin, a_cos = box_muller(w[0], w[1])
        b_sin, b_cos = box_muller(w[2], w[3])
        return a_sin, a_cos, b_sin, b_cos
    return (
        trunc_normal_from_uniform(uniform(w[0])),
        trunc_normal_from_uniform(uniform(w[1])),
        trunc_normal_from_uniform(uniform(w[2])),
        trunc_normal_from_uniform(uniform(w[3])),
    )


def values_at(seed: int, threads: int, positions: I64, offset_calls: int = 0, dist: Dist = "normal") -> F32:
    """Weights (std units) PyTorch-CUDA would write at flat ``positions``.

    ``offset_calls`` is the generator's Philox offset divided by 4, i.e. how
    many 4-value draws per thread earlier fills already consumed.
    """
    thread = positions % threads
    k = positions // threads
    call = k // 4 + offset_calls
    comp = k % 4
    w = philox4x32_10(
        (call & 0xFFFFFFFF).astype(np.uint32),
        (call >> 32).astype(np.uint32),
        (thread & 0xFFFFFFFF).astype(np.uint32),
        (thread >> 32).astype(np.uint32),
        seed & 0xFFFFFFFF,
        (seed >> 32) & 0xFFFFFFFF,
    )
    return np.choose(comp, list(block_values(w, dist))).astype(np.float32)


def normal_at(seed: int, threads: int, positions: I64, offset_calls: int = 0) -> F32:
    """Standard normals ``Tensor.normal_`` on CUDA would write at flat ``positions``."""
    return values_at(seed, threads, positions, offset_calls, "normal")
