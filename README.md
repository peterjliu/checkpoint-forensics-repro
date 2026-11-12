# checkpoint-forensics-repro

An independent reproduction of the core result in Will DePue's report
[Top Chinese Frontier AI Models Carry the Fingerprint of Export-Controlled NVIDIA Blackwell GPUs](https://unemploymentlabs.com/checkpoint-forensics)
(Unemployment Labs, November 12, 2026).

Everything here runs on a CPU and uses only weights published on Hugging Face.
No GPU is needed: the GPU random-number layout is emulated in numpy.

## The idea

PyTorch fills a large tensor on a GPU with `SMs x 2048` threads
([`DistributionTemplates.h`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DistributionTemplates.h)).
Thread `i` draws from Philox stream `i` of the seed and writes its k-th value
to flat position `i + k * threads`.

```
threads = SMs * 2048          132 SMs -> 270,336   148 -> 303,104   152 -> 311,296

position p  ->  thread = p % threads
                k      = p // threads      (k-th value that thread drew)
                value  = philox(seed, stream=thread, call=k // 4)[k % 4]
```

So the same seed puts the same numbers in different places on chips with
different SM counts. Embedding rows for tokens that never occur in training
(for example the 13 bytes that cannot appear in valid UTF-8) receive no
gradient and keep their starting values, so the layout can be read back from
the released weights.

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
