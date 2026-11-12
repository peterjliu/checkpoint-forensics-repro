"""Download one named tensor's raw bytes from a Hugging Face safetensors shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forensics.remote import download_tensor, file_url, read_header


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo")
    ap.add_argument("filename")
    ap.add_argument("tensor")
    ap.add_argument("dest", type=Path, help="output .bin path; a .json sidecar holds dtype/shape")
    args = ap.parse_args()
    url = file_url(args.repo, args.filename)
    info = next(t for t in read_header(url) if t.name == args.tensor)
    download_tensor(url, info, args.dest)
    meta = {"repo": args.repo, "file": args.filename, "tensor": info.name, "dtype": info.dtype, "shape": list(info.shape)}
    args.dest.with_suffix(".json").write_text(json.dumps(meta))
    print(f"wrote {args.dest} {info.dtype} {info.shape}")


if __name__ == "__main__":
    main()
