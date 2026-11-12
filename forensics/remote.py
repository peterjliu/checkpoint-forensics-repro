"""Read safetensors headers and tensor byte ranges from Hugging Face over HTTP.

Only the bytes needed are fetched, so a multi-hundred-GB checkpoint never has
to be downloaded in full.
"""

from __future__ import annotations

import json
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_UA = "checkpoint-forensics-repro/0.1"


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int  # absolute byte offset in the file
    end: int

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


def file_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def fetch_range(url: str, start: int, end: int) -> bytes:
    """Bytes [start, end) of ``url``."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}", "User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data: bytes = resp.read()
    if len(data) != end - start:
        raise IOError(f"short read from {url}: wanted {end - start} bytes, got {len(data)}")
    return data


def read_header(url: str) -> list[TensorInfo]:
    (n,) = struct.unpack("<Q", fetch_range(url, 0, 8))
    raw: dict[str, dict[str, object]] = json.loads(fetch_range(url, 8, 8 + n))
    out: list[TensorInfo] = []
    for name, meta in raw.items():
        if name == "__metadata__":
            continue
        offsets = meta["data_offsets"]
        shape = meta["shape"]
        dtype = meta["dtype"]
        assert isinstance(offsets, list) and isinstance(shape, list) and isinstance(dtype, str)
        out.append(TensorInfo(name, dtype, tuple(int(s) for s in shape), 8 + n + int(offsets[0]), 8 + n + int(offsets[1])))
    return sorted(out, key=lambda t: t.start)


def download_tensor(url: str, info: TensorInfo, dest: Path, chunk: int = 256 << 20) -> Path:
    """Download one tensor's raw bytes to ``dest`` (skipped if already complete)."""
    size = info.end - info.start
    if dest.exists() and dest.stat().st_size == size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as f:
        for lo in range(info.start, info.end, chunk):
            f.write(fetch_range(url, lo, min(lo + chunk, info.end)))
    tmp.rename(dest)
    return dest


def decode(raw: bytes | NDArray[np.uint8], dtype: str) -> NDArray[np.float32]:
    """Decode BF16 / F16 / F32 safetensors bytes to float32."""
    buf = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, bytes) else raw
    if dtype == "BF16":
        return (buf.view(np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)
    if dtype == "F16":
        return buf.view(np.float16).astype(np.float32)
    if dtype == "F32":
        return buf.view(np.float32)
    raise ValueError(f"unsupported dtype {dtype}")
