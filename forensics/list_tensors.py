"""List the large float tensors in one safetensors shard on Hugging Face."""

from __future__ import annotations

import argparse

from forensics.remote import file_url, read_header


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo")
    ap.add_argument("filename")
    ap.add_argument("--min-numel", type=int, default=1_000_000)
    ap.add_argument("--grep", default="")
    args = ap.parse_args()
    infos = read_header(file_url(args.repo, args.filename))
    print(f"{len(infos)} tensors in {args.repo}/{args.filename}")
    for t in infos:
        if t.numel >= args.min_numel and args.grep in t.name:
            print(f"{t.dtype:8s} {str(t.shape):24s} {t.numel:>14,d}  {t.name}")


if __name__ == "__main__":
    main()
