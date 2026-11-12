# checkpoint-forensics-repro

An independent reproduction of the core result in Will DePue's report
[Top Chinese Frontier AI Models Carry the Fingerprint of Export-Controlled NVIDIA Blackwell GPUs](https://unemploymentlabs.com/checkpoint-forensics)
(Unemployment Labs, November 12, 2026).

Everything here runs on a CPU and uses only weights published on Hugging Face.
No GPU is needed: the GPU random-number layout is emulated in numpy.

## The concept

### 1. Weights start as seeded random numbers

Before training, every weight is set to a random number. The numbers come from
a pseudorandom generator (Philox, via NVIDIA's cuRAND), which is fully
determined by a **seed**: same seed, same numbers, every time.

### 2. The GPU decides where each number lands

To fill billions of weights quickly, PyTorch splits the job across GPU threads.
Each thread has its own stream of numbers from the seed, and the threads take
turns across the tensor. Here is a toy GPU with 6 threads filling 18 weights.
The letter is the thread, the digit is how many numbers it has drawn so far:

```
6 threads:   A0 B0 C0 D0 E0 F0 | A1 B1 C1 D1 E1 F1 | A2 B2 C2 D2 E2 F2
position:     0  1  2  3  4  5    6  7  8  9 10 11   12 13 14 15 16 17
```

Thread A's numbers land at positions 0, 6, 12. Now fill the same tensor with
the same seed on a GPU with 5 threads:

```
5 threads:   A0 B0 C0 D0 E0 | A1 B1 C1 D1 E1 | A2 B2 C2 D2 E2 | A3 B3 C3
position:     0  1  2  3  4    5  6  7  8  9   10 11 12 13 14   15 16 17
```

Thread A draws exactly the same numbers, but they now land at 0, 5, 10. Only
the first pass (positions 0 to 4) agrees between the two; everything after is
shuffled. The arrangement of the numbers records how many threads filled them.

### 3. The thread count is set by the chip

PyTorch does not pick the thread count; the hardware does. For a large fill it
uses 2,048 threads per streaming multiprocessor (SM), the GPU's basic compute
unit ([`calc_execution_policy` in `DistributionTemplates.h`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DistributionTemplates.h)):

| SMs | Threads | Chips reported with this SM count |
|---|---|---|
| 108 | 221,184 | A100 / A800 |
| 132 | 270,336 | H100 SXM / H800 / H200 (Hopper) |
| 148 | 303,104 | B200 / B300 (Blackwell), per the report |
| 152 | 311,296 | GB200 / GB300 (Blackwell), per the report |

The exact rule, which `forensics/philox.py` implements:

```
threads = SMs * 2048

position p  ->  thread = p % threads
                k      = p // threads      (k-th number that thread drew)
                value  = philox(seed, stream=thread, call=k // 4)[k % 4]
```

### 4. Some weights are never trained

Training normally overwrites the starting values. But a token embedding row
only changes when its token appears in the training data, and some tokens never
do. The clearest case is the 13 byte values that cannot occur in valid UTF-8
(`0xC0`, `0xC1`, `0xF5` to `0xFF`): their rows receive no gradient and still
hold their starting numbers in the released model. Reserved and unused special
tokens behave the same way.

### 5. Reading the fingerprint back

Given a model's published embedding table:

1. Find rows that look untouched (pure random noise, all with the same spread).
2. For a candidate seed and SM count, regenerate what those positions would
   have held, and compare with the published values.
3. The right seed with the right SM count reproduces the rows almost exactly
   (correlation above 0.99). The right seed with the wrong SM count, or the
   wrong seed, gives no correlation at all.

If the seed is unknown, try all 2^32 of them: a handful of untouched values is
enough to single out one seed, because a wrong seed has no realistic chance of
matching several values at once.

The emulation is a few dozen lines of numpy and runs on a CPU, so the claim can
be checked by anyone without access to the chips in question.

## Results

Each model's embedding table was scored against every SM count from 100 to 160.
Every model has exactly one peak; all other layouts score under 4 sigma.
Sigma is `r * sqrt(values compared)`, the same definition the report uses.

| Model | Seed | Seed source | Matching layout | Untouched rows | r | sigma |
|---|---|---|---|---|---|---|
| DeepSeek-V3-Base | 953049583 | report | 132 | 489 | 0.999999 | 1,872 |
| DeepSeek-V4-Flash | 953049583 | report, and recovered blind | 148 | 461 | 0.99987 | 1,377 |
| DeepSeek-V4-Pro | 953049583 | report | 152 | 468 | 0.99990 | 1,831 |
| Kimi K3 | 20256586 | report (offset recovered here) | 152 | 16 | 0.999996 | 339 |
| Kimi K3.1-Pro | 20256586 | report | 152 | 14 | 0.999995 | 317 |
| MiniMax-M3 | 2760 | recovered blind | 148 | 14 | 0.99999 | 293 |
| Ling-3.1-flash | 3952 | recovered blind | 152 | 1,154 | 0.983 | 2,138 |

Checks that the matches are not artefacts:

- **Blind seed recovery.** `forensics.seed_search` tries all 2^32 seeds against a
  few untouched values without being told the answer. DeepSeek-V4-Flash returns
  one hit: the report's seed, at 148 SMs. The MiniMax-M3 and Ling-3.1 seeds are
  not published in the report; each search returns a single hit, at the layout
  the report claims.
- **Seed off by one** gives zero matching rows on every model.
- **Kimi K3** matches only at a generator offset of 944 draws, found by
  searching 8.4M offsets x 61 layouts (one hit).
- **Ling-3.1** matches only when modelled as `trunc_normal_` cut at 3 std; the
  0.98 correlation is the figure the report gives.
- **Bit-exactness.** DeepSeek-V3 is 99.9% bit-identical to the CPU emulation and
  V4-Flash 95%; the remaining values differ in the last bit (CPU vs GPU `log`,
  `sin`, `cos`).

## What this does not show

- **Which chip has which SM count.** That 148 SMs means B200/B300 and 152 means
  GB200/GB300 is taken from the report. It was not checked on hardware here, and
  public spec pages disagree.
- The weaker "partial survival" signal in trained layers, the NVIDIA-vs-AMD
  numerical test, and the GPU-count and cost estimates were not attempted.
- Apart from DeepSeek-V3, the Hopper-era models and the 24 known-hardware cases
  were not checked.
- As the report says, the fingerprint shows what hardware class initialised the
  weights. It does not show where the chips were, who owned them, or that the
  whole training run used them.

## Run it

Needs [`uv`](https://docs.astral.sh/uv/) and about 12 GB of disk. Only the
embedding tables are downloaded, by HTTP range request.

```
make test     # emulator known-answer tests + search/scan tests
make fetch    # download the seven embedding tables into data/
make scan     # score SM counts 100..160 for each model
make blind    # recover seeds from the weights alone (about 10 min each on 14 cores)
```

## Layout

| File | Purpose |
|---|---|
| `forensics/philox.py` | Philox4x32-10 and PyTorch-CUDA `normal_` / `trunc_normal_` placement, in numpy |
| `forensics/remote.py` | Read safetensors headers and single tensors over HTTP |
| `forensics/scan.py` | Score candidate SM counts for a known seed |
| `forensics/seed_search.py` | Search seeds, or generator offsets, from untouched values |

## License

Apache 2.0. See [LICENSE](LICENSE).
