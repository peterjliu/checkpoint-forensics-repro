# checkpoint-forensics-repro

CPU-only reproduction of GPU-layout fingerprints in published model weights. See README.md for method and results.

- `uv` for Python; run tools through the Makefile (`make test`, `make check`, `make fetch`, `make scan`, `make blind`).
- Downloaded weights go in `data/` (git-ignored). Run code that reads them with `python -I`, as the Makefile does.
- Tests first; keep pyright clean (`make check`).
- When a new model is reproduced, add its command to the Makefile and its row to the README results table.
