# Split attention: GPU and CPU in parallel over one KV cache (design note, 2026-10-28)

The idea, from David Sandquist (2026-10-28): the bus is the bottleneck, so put a co-processor on the host side, next to the memory, that
preemptively sorts and compresses the store and hands back small things (hashes, results) instead of the full data.
The CPU is that co-processor: it sits next to the host KV store, idle while the GPU waits for PCIe. The idea has three
roles, one design; this note works out the first and keeps the other two in view:

1. During a token, attend in place: the CPU attends over cold pages and returns partial results, merged exactly on
   the GPU (this note; exact).
2. Between tokens and while idle, consolidate: rank pages by use, reorder them so pages that are read together lie
   together, keep the hot ones in the GPU pool, grade precision by importance (placement is exact; graded precision
   is lossy). Overlaps `tiered-kv-scheduler.md`.
3. Index: per-page hashes or summaries, later a learned index (a small model that predicts from the query which pages
   matter), first for prefetch order (exact), later for skipping pages (approximate, needs quality tests).

## Why it is exact

Attention over a set of pages gives, per query row and head, an output vector and its log-sum-exp (LSE). Results over
disjoint page sets merge exactly: with LSE values l1, l2 and outputs o1, o2, the merged output is
(e^l1 o1 + e^l2 o2) / (e^l1 + e^l2) and the merged LSE is log(e^l1 + e^l2). Flash-decoding's split-K combine is the
same thing. No approximation, so no quality question: the probe must give identical answers.

## Why it could pay (measured on the test machine (RX 9070) 2026-10-28, not yet as a split)

- PCIe route: the stream at 250K (Saluki, 2420 MiB pool) moved about 6.2 GB of KV per token at 3.9 t/s, about 24 GB/s.
- CPU route: `--no-kv-offload` at 125K processed about 4.35 GB of KV per token at 5.4 t/s, about 23 GB/s (helped by
  n-gram drafting, so an upper estimate).
- Bus traffic for the CPU's share: per token and attention layer the query (24 heads x 256 floats) goes down and the
  partial output plus LSE comes back, about 0.4 MB per token for all 16 layers, instead of gigabytes.
- Two routes of similar capacity used one at a time today; used in parallel with the pages split in proportion,
  writing at depth could roughly double. Arithmetic from two measurements, to be measured.
- Catch: both routes read the same dual-channel DDR4 (the GPU's PCIe reads are DMA from host RAM too; DDR4-3200
  dual channel peaks at about 51 GB/s). 24 + 23 GB/s is close to that peak, so the gain may be nearer 1.5x than 2x.
  The microbenchmark should run the CPU attention while a stream is active, not only alone.

## First measurement (2026-10-28): CPU attention alone

`test-backend-ops perf -b CPU -o FLASH_ATTN_EXT`, new perf case: one query row, Qwen3.8/Saluki shape (4 KV heads x 256,
GQA 6), q8_0 K and V, cache larger than the 64 MB L3, nothing else running:

| cells | time per layer | KV read | rate | compute |
|---|---|---|---|---|
| 65,536 | 10.0 ms | 143 MB | 14.2 GB/s | 161 GFLOPS |
| 262,144 | 39.5 ms | 570 MB | 14.4 GB/s | 163 GFLOPS |

So the CPU route is 14 GB/s today, below the 23 GB/s the no-kv-offload run suggested (that included drafting). It is
compute-bound, not memory-bound: 163 GFLOPS is about a tenth of what 12 Zen 3 cores can do with AVX2 FMA, so the CPU
kernel has room (e.g. sharing each K/V row across the 6 query heads of its group, q8 dot products in integer SIMD).

Arithmetic for Saluki at 250K with the 2420 MiB pool (6.2 GB streamed per token, 3.9 t/s now): split in proportion to
24 GB/s (PCIe) and 14 GB/s (CPU), the streamed part takes about 160 ms instead of 256 ms, so about 6 t/s, about 1.5x.
With a CPU kernel at 25 GB/s it would be about 2x, if DDR4 has the headroom when both run (to measure).

Grouped-query decode path (f66e68626, same day): one work item per (KV head, chunk) serves all 6 query heads of the
group, so each K/V row is read and converted once: 65,536 cells 6.3 ms (22.6 GB/s), 262,144 cells 26.4 ms (21.6 GB/s),
244 GFLOPS, 1.5x. What is left is the heads' own arithmetic (6 dot products and 6 multiply-adds of 256 per cell);
next would be blocks of cells (scores for 32 cells, then one softmax step), at about 15% of peak now.

## Contention (2026-10-28, `kvstream/bench/run-contention.sh`, logs `results/rx9070-ryzen5900x/contention-2026-10-28-*.log`)

Saluki at 150K (2420 MiB pool, about 2.7 GB streamed per written token, no drafting) while the CPU benchmark above runs
with all threads:

| | stream writing | CPU attention, one 256K layer pass |
|---|---|---|
| each alone | 8.9 t/s (24 GB/s over PCIe) | 47 ms (12 GB/s) |
| both at once | 8.0 t/s (21.6 GB/s) | 65 to 68 ms (8.6 GB/s) |

Together about 30 GB/s of KV against 24 for the stream alone: the routes slow each other (CPU about 30%, stream about
10%). Split in proportion to the contended rates, a token at 150K would take about 89 ms instead of 112: about 1.25x
with the first CPU kernel. Not yet separated: DDR4 bandwidth against CPU time (the benchmark used every thread, and the
stream's copier threads need CPU too), and how far a faster CPU kernel moves it. With only the server loaded and idle,
the CPU pass took 47 to 50 ms instead of 39.5 ms on an empty machine: the server's threads take CPU while idle.

Rerun with the grouped-query CPU path (log `results/rx9070-ryzen5900x/contention-2026-10-28-gqa-kernel.log`):

| | stream writing | CPU attention, one 256K layer pass |
|---|---|---|
| each alone | 8.9 t/s (24 GB/s) | 29 ms (19.6 GB/s) |
| both at once | 7.8 t/s (21 GB/s) | 40 to 42 ms (13.9 GB/s) |

Together about 35 GB/s (24 for the stream alone, 30 with the first kernel). Split at these contended rates, a token at
150K would take about 77 ms instead of 112: about 13 t/s instead of 8.9, about 1.45x. Arithmetic from rates measured
under the load the split creates; the split itself is the measurement still to come.

## The split, built (2026-10-29, branch `split-attention`)

How it runs (`LLAMA_KV_SPLIT=fraction`, off by default; `LLAMA_KV_SPLIT_THREADS`, default 8):

- In each decode ubatch (up to 96 query tokens) the stream gives the CPU the oldest pages that are not resident, that
  fraction of the pages that would stream, while they form one block from the first page. Those pages are not
  copied, and the GPU's page table starts after them (its mask is a view of the last columns).
- The HIP paged decode op copies Q to shared host memory and raises a flag, attends over its pages with its log-sum-exp,
  waits for the CPU's flag, and merges (`pd_split_q`, `pd_split_signal`, `pd_split_wait`, `pd_split_merge`).
- A CPU worker thread in the stream waits for each layer's Q, runs ggml's CPU flash attention (grouped-query path,
  log-sum-exp output) over the block in place in the host store, and raises its flag.
- The op carries the hand-over as `ggml_flash_attn_ext_set_split` (op params grew from 64 to 80 bytes for the pointer);
  the step tag input has a second entry for it. Backends that cannot merge refuse the op, and the stream then turns the
  split off with a warning (CUDA today; the CPU half is portable as it is).

First measurements (Saluki, 150K, pool 2420 MiB, no drafting, `kvstream/bench/run-split.sh` and `run-split-sweep.sh`,
logs `results/rx9070-ryzen5900x/split-*-2026-10-29.log`):

| CPU share : threads | write t/s | CPU attends per token | CPU waits for Q | CPU rate |
|---|---|---|---|---|
| no split | 8.9 | | | |
| 0.2 : 8 | 9.2 | 68 ms | 27 ms | 9.0 GB/s |
| 0.2 : 12 | 10.1 | 67 ms | 27 ms | 9.1 GB/s |
| 0.3 : 16 | 8.1 | 94 ms | 22 ms | 9.9 GB/s |
| 0.4 : 8 | 6.4 | 130 ms | 22 ms | 9.5 GB/s |
| 0.4 : 12 | 6.0 | 133 ms | 22 ms | 9.3 GB/s |

Exact in practice: with 0.4 : 8, the three 400-token greedy answers were word for word identical to the unsplit ones,
and the code word was found in every run. Fastest so far 0.2 : 12, +13%. The CPU is the critical path: it attends at
about 9.5 GB/s whatever the thread count, against 13.9 GB/s for the same kernel beside a stream in the contention
benchmark and 21.6 alone. Differences from the benchmark, to measure next: the host store interleaves the 4 KV heads in
each row (the grouped path's work items are (KV head, chunk), so each reads a quarter of every row), and the work comes
as 16 small passes per token (17,664 rows per layer at 0.2) with the pool woken for each. The waits for Q (22 to 27 ms
per token) are the GPU's other layers, when the CPU has nothing to do.

## After the first measurements (2026-10-29 afternoon)

Steps, each its own commit on `split-attention` with logs in `results/rx9070-ryzen5900x/`:

| step | CPU alone, 1 / 5 query tokens (fa-cpu-layout) | Saluki 150K, no drafting | Qwen3.8 125K, MTP |
|---|---|---|---|
| first split | 15 / 4.5 GB/s | 10.1 t/s (0.2 : 12) | |
| whole rows per work item | 19.6 / 4.5 | 11.4 (0.3 : 16) | fixed 0.2: 15.9 (none 33.8) |
| blocked kernel (16 rows) | 25 / 7.3 | 11.5 (0.3) | fixed 0.1: 31.7 |
| batched q8_0 scores, vector exp | 27 to 36 / 8.3 to 9.9 | 11.7 (0.4), 11.3 (0.3) | fixed 0.2: 19.6, 0.3: 23.1 (none 32.6) |

- Auto (time per token kept, steps of 0.1, start at no split): settles at 0.3 for Saluki (106 -> 83 ms per token) and
  stays at no split with MTP (33.6 t/s against 32.6 to 34.0 without). 0.05 steps got stuck at 0, since a small share
  does not cover the split's fixed cost per ubatch.
- No regression where nothing streams: at 30K (cache in the pool) auto never splits (33.8 against 33.7 to 33.9).
- Ceiling (LLAMA_KV_SPLIT_NOOP, the CPU part skipped, output wrong): MTP at 125K 39.4 t/s at 0.3, 40.2 at 0.5
  against 33.7, with fewer drafts accepted, so roughly 1.3 to 1.4x per verify pass; Saluki 12.6 at 0.3, 16.3 at 0.5.
- So without drafting the split gains about 30%, close to what the shared DDR4 allows (about 42 GB/s of both routes
  against about 51 GB/s peak). With drafting the CPU part, 30 query vectors per KV head, is bound by arithmetic and
  would need 2 to 3x more in the server; not pursued further for now.

## Prior art (survey 2026-10-28, `~/code/scratch/llm-perf-investigation/research-kv-offload-2026-10-28.md`)

Closest: HGCA (arXiv 2507.03153: GPU dense over recent KV, CPU sparse over the rest, LSE merge; no code found),
JazzyJJ0601/attention-offload (exact split, PyTorch, RTX 3090 Ti, no streaming tier), Samsung's CXL near-memory demo
(hot pages on the GPU, attention results from the memory device). Nothing in upstream llama.cpp. Apparently new here:
an exact split inside ggml/llama.cpp, combined with a streaming tier, on a consumer AMD card.

## What exists in the fork

- GPU merge by LSE: the paged prefill runs stock attention over chunks and merges them by LSE
  (`fattn-paged-decode.cu`, `pd_part`, commits 9e9e46f86 and 189fb51c0; `GGML_FA_PARAM_LSE`).
- CPU attention over paged K/V: `ggml_compute_forward_flash_attn_ext_paged` (`ggml-cpu/ops.cpp`), the test reference.
  It gathers the pages into a contiguous copy first: fine for tests, too slow for the hot path.

## What is missing

1. Done 2026-10-28 (d00c35b45): CPU attention writes its LSE, all three CPU paths. Still to do: read pages in place
   instead of gathering them.
2. Done 2026-10-28 (62e790a52): the paged decode kernel writes its LSE. `tests/test-fa-lse` checks the split itself on
   the RX 9070: the GPU's part merged with the CPU's part is within 3.3e-4 of the CPU's full result (q8_0, one row),
   the level of the ordinary GPU/CPU difference.
3. Per layer at decode: split the page list into GPU pages (pool + streamed) and CPU pages (cold, in the host store),
   run both, copy the CPU's output and LSE to the GPU, merge. The CPU part must start as soon as the layer's Q exists
   and overlap the GPU part; ggml_backend_sched launches GPU splits asynchronously, so ordering the GPU node first
   may give the overlap without new threading. To be checked.
4. A split ratio from measured rates (PCIe GB/s vs CPU GB/s), later adaptive per step.

Related: `tiered-kv-scheduler.md` (which pages live where); the research survey in
`~/code/scratch/llm-perf-investigation/research-kv-offload-2026-10-28.md` (prior art).
