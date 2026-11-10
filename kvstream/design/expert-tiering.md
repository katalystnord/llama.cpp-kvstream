# Expert tiering: MoE experts paged like the KV cache

Written 2026-11-09, evening, while the card ran an extraction. Nothing here has run on a GPU yet. Every claim about
other people's work was checked against their code or posts tonight; numbers quoted from them are theirs, not ours.

## The idea

The KV stream keeps a bounded working set of KV pages in VRAM and the rest in host RAM. A MoE model can be handled
the same way: routing is skewed, so a small hot set of experts serves most routed reads. Keep the hot experts in
VRAM, keep the rest in host RAM (computed on the CPU, or copied in), and size the hot set from the memory that is
actually free. Nothing is thrown away, so quality is unchanged; only speed depends on the budget.

Started from MAPLE (github.com/AmSach/maple), which does the lossy version of this: drop the cold experts for good.

## MAPLE, as checked

What it does: run `llama-imatrix` on a calibration text, score each expert by routing count times sqrt(mean input
energy), keep the top 48 of 128 per layer of Gemma 4 26B-A4B, and rewrite the GGUF with only those experts (the
router rows are sliced the same way). 4.75 GB, fits a 6 GB laptop GPU, 45 t/s against 6.9 t/s offloaded.
Apache-2.0. About 230 lines of real logic.

What the repository shows:

- The profiler reads `<tensor>.imatrix.call_counts`. Stock `llama-imatrix` writes `<tensor>.counts` and
  `<tensor>.in_sum2` (checked in this fork and in upstream master tonight), so the script does not run on a stock
  imatrix as published. The README's `--imatrix` / `--target-experts` / `--base-model` options do not exist in the
  scripts; paths and the 30 x 128 shape are hard-coded (Windows `D:\` paths).
- The calibration text is 1,979 tokens. The README's own figure, 79.88% of activation mass kept, means about one
  routed read in five on the calibration text itself went to an expert that was removed.
- The quality evidence is four prompts scored by keyword hits plus one assert. "100% quality retention" rests on that.
- Not checked: the published GGUF itself (not downloaded).

The careful version of expert pruning is REAP (Cerebras, ICLR 2026): saliency from router gate values times expert
output norms, evaluated on 20B to 1T models, with code and pruned checkpoints published. If pruning is ever wanted,
start there, not from MAPLE.

What is worth taking from MAPLE: the hot set can be read from a stock imatrix for free (`.counts` per expert), and
many GGUF repositories already ship one. That feeds placement, which is lossless, instead of pruning.

## Prior work on lossless tiering in llama.cpp

**Correction, 2026-11-10 morning: upstream has merged an expert cache.** The first version of this note said none
was merged; that came from reading the RFC threads, not upstream's history, and was wrong.

- PR #29887 (am17an, merged 2026-10-07), "llama : add a GPU cache for MoE experts kept in host memory": a port of
  tetherto/qvac-fabric's cache. `MUL_MAT_ID` on host-resident experts runs on the GPU through an LRU cache of experts
  per layer; only misses are uploaded. Small batches only (<= 32 tokens); larger batches use the normal offload path.
  Enable with `--moe-cache-mib N` (they found about 10% of the total expert size a good start). Reported +57% to
  +120% decode on Qwen3.8-Flash-Next with an RTX 4090 / 5090.
- PR #30112 (merged 2026-10-08): multiple GPUs. PR #30637 (2026-10-19): the LRU replaced with a 50/50 SLRU.
- It lives in `src/llama-moe-cache.cpp` (backend-agnostic: any GPU or iGPU device through the scheduler's copy
  callback), so it is not CUDA only. Both forks contain it: brick's stream fork (bcf1a81d6, the 2026-10-08 LRU
  version) and beast's (a37cc4610, with the SLRU). Not yet tried on HIP.
- Design difference from leloch: upstream uploads misses and computes all experts on the GPU; leloch computes misses
  on the CPU and never waits for an upload. Which is better depends on PCIe speed and expert size.

Other work, not merged:

- **Issue #20757** (Mar 2026, closed without a merge): two-tier GPU slot cache plus pinned RAM, from a Python proof
  of concept. In the comments, lee-b proposes using imatrix counts to pick the hot experts.
  martinalderson built a proof of concept **on an RX 9070 XT with Vulkan**, Gemma 4 26B-A4B: dynamic LRU, 8 slots
  per layer per tensor, 27.9 -> 35.1 t/s decode (+26%), hit rate 47% at 100 tokens rising to 67% at 500+. Their key
  finding: in the Vulkan backend, `set_tensor_async` from mmap'd memory is synchronous (staging buffer plus
  synchronise), and without their SAM/ReBAR direct-write bypass (#21590) the GPU path was slower (9 against 28 t/s).
  Code: martinalderson/llama.cpp, branch `moe-profile` (they call it messy), with an activation profiler.
- **RFC discussion #24528** (leloch, from Jun 2026): the most complete work. Branch `leloch/llama.cpp`
  `moe-cache-v2-pr`, about 9,000 added lines, CUDA only (other backends get no-op hooks). Design: `MUL_MAT_ID` stays
  on the CPU; experts that are resident run on CUDA, misses run on the CPU at the same time; misses are filled in the
  background for later use, never waited on. Reported +7% to +57% decode on multi-3090 rigs, others +30% and +72%,
  but a regression on a GTX 1080 Ti (19.3 -> 13.3 t/s) and a prefill loss of 14% to 66% on one RTX 5090 setup.
  No maintainer replies in the visible thread. Its own limitations list: CUDA only, demand fill only, no prefetch,
  **no hot-set file, no persistence across restarts**.
- Papers: Fiddler (CPU computes cold experts instead of copying them), MoE-Infinity (activation tracing, prefetch,
  caching), HOBBIT (mixed precision for cold experts, built on llama.cpp), MoBiLE (big/little experts on consumer
  GPUs), TriMoE (argues hot / warm / cold, not two tiers).

## Where we could add something

1. **HIP / RDNA 4.** First try upstream's backend-agnostic `--moe-cache-mib`, which beast's fork already has; a
   port of leloch's cache only matters if its CPU-computes-misses design beats it. leloch's code is CUDA; ggml's HIP build compiles `ggml-cuda/*.cu`, so a
   port may be smaller than it looks, but the compute capability gate (HIP uses offset values) and the fill stream
   need care. Our copy engine already knows the SDMA path, and also its deadlock ([copy-stall.md](copy-stall.md)):
   expert fills on the same queue would meet the same problem, so this comes after the stall fix.
   What leloch's `ggml-cuda/moe-cache.cu` (3,145 lines) needs from the runtime, checked tonight: only plain
   runtime calls (`cudaMalloc`/`Free`, `cudaMallocHost`/`FreeHost`, `cudaMemcpy(Async)`, streams with
   `cudaStreamNonBlocking`, `cudaStreamCreateWithPriority`, `cudaDeviceGetStreamPriorityRange`, `cudaMemGetInfo`,
   `cudaGetDeviceProperties`, `cudaStreamSynchronize`). No graphs, no VMM, no peer access. For a HIP build, ggml's
   `vendors/hip.h` maps all of these except the two priority calls (add `hipStreamCreateWithPriority` and
   `hipDeviceGetStreamPriorityRange` defines). The other gate is `cuda_device.cc < min_compute_capability`
   (line 946): on HIP, `cc` carries an AMD offset, so the check needs an AMD branch or `GGML_CUDA_MOE_CACHE_MIN_CC`.
   The same list is what ZLUDA would have to support to run a CUDA build of the branch unchanged on the RX 9070
   (ZLUDA is being looked at separately), and its reported compute capability decides whether the gate passes.
2. **Warm start from an imatrix hot set.** Fills the cold-start gap (47% hit rate at 100 tokens in the RX 9070 XT
   test) and is on leloch's list of things it lacks. Small and separable: `scripts/expert-hotset.py` already writes
   the hot set per layer for a given budget.
3. **One VRAM budget for KV pages and expert slots.** The stream already holds a fixed resident pool. With a MoE
   model the same pool could be split between KV pages and expert slots by which pays more per byte, shifting as the
   context grows. This is the "optimise towards available memory on the fly" part, and nobody else couples the two.
   Needs a MoE model the paged kernels support (today: head size 256, 6 query heads per KV head).

## Tomorrow: test on brick first

brick (see [../second-machine/MACHINE.md](../second-machine/MACHINE.md)): GTX 1650 Max-Q 4 GB, sm_75, CUDA 12.4,
32 GB DDR4-2666 dual channel, PCIe 3.0. Why it is the right first target:

- leloch's cache runs there today with no porting: `--moe-cache on` and fixed budgets need compute capability 7.0
  or newer (sm_75 qualifies); only `auto` needs two 8.0+ devices.
- It is the hard case: a 35 W GPU on PCIe 3.0, close to the class where the 1080 Ti regressed. A gain on brick
  means something; a loss tells us where the floor is.
- The RX 9070 stays free for its own work.

Model: gpt-oss-20b (already on disk, `~/models/gguf/gpt-oss-20b-unsloth/gpt-oss-20b-Q8_0.gguf`, 12.1 GB). Checked
tonight: dense weights 1.79 GiB (fit in 4 GB), experts 9.48 GiB in MXFP4, 24 layers, 32 experts, 4 used per token,
4.2 MiB per expert per tensor (above the cache's 1 MiB floor for pre-8.0 devices; MXFP4 has a CUDA matvec).
Estimate, not measured: about 1 to 1.5 GiB left for the cache after dense weights, KV and compute buffers.

Watch out: the cache's default reserve is `free VRAM - 3072 MiB`, which leaves nothing on a 4 GB card. Lower it with
`GGML_CUDA_MOE_CACHE_RESERVE_MB` (for example 256).

Steps:

1. Build leloch's branch for sm_75 on brick:
   ```
   git clone --branch moe-cache-v2-pr https://github.com/leloch/llama.cpp llama.cpp-moe-cache
   cmake -B build-sm75 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
   cmake --build build-sm75 --target llama-bench llama-server llama-imatrix test-moe-cache --parallel 3
   CUDA_VISIBLE_DEVICES=0 ./build-sm75/bin/test-moe-cache
   ```
2. Isolated A/B, same CPU-resident weights in both arms (from their doc; budgets in MiB):
   ```
   GGML_CUDA_MOE_CACHE_RESERVE_MB=256 ./build-sm75/bin/llama-bench -m gpt-oss-20b-Q8_0.gguf \
     -p 0 -n 300 --n-gen-warmup 256 -r 3 -ngl 99 --cpu-moe -t 6 -fa on \
     --moe-cache off,512,1024 --repack off -v -o json
   ```
   Also the stock best: `--moe-cache off --repack on` with the best `--n-cpu-moe`. Check the log for
   `[moe-cache] enabled` and the teardown counters (hits, misses, fills). Record prefill too (pp512).
3. Routing skew for our own work: `llama-imatrix` on gpt-oss-20b over a text like what we really run (agent and
   coding transcripts, not wiki text), then
   `kvstream/scripts/expert-hotset.py imatrix.gguf --model gpt-oss-20b-Q8_0.gguf --budget-mib 512 1024 1536`.
   That gives the best static hit rate for each budget, to compare with the cache's measured rate.
4. Decide: if brick gains, the HIP port (1) and the warm start (2) are worth doing; if it loses, find out whether
   the CPU, PCIe 3.0 or the GPU is the limit before porting anything.

On the RX 9070 later: Qwen3.6-35B-A3B is the natural model (Q4_K_M 20.6 GiB does not fit 16 GB; on 2026-10-02 the
best stock setting was `--n-cpu-moe 16` at 31.2 t/s decode, Vulkan, `~/models/llama-models/bench-35b.txt`). Only the Q8_0
file (36.9 GB) is on disk now.

## Results on brick, night of 2026-11-09

Full record: brick, ~/code/moe-cache-brick/DOCUMENTATION.md (commands, logs, all numbers). leloch's branch at
e3096b0, built for sm_75 with g++-13 as the CUDA host compiler; gpt-oss-20b MXFP4; GGML_CUDA_MOE_CACHE_RESERVE_MB=256.

- test-moe-cache: 36 OK, 2 SKIP (need two GPUs or a dynamic backend), 1 FAIL (cache-capabilities: its auto section
  assumes a compute capability 8.0 card; brick's 7.5 is correctly refused by auto).
- llama-bench, random tokens, all experts on the CPU: cache off 16.40 t/s; 1024 MiB 23.17 (+41%); 1900 MiB 26.74
  (+63%). Best stock setting (-ncmoe 22, repacking on): 16.37. Prefill unchanged (about 97 t/s).
- Real text through llama-server (3 code and docs prompts, 400 tokens each): stock best 16.16, cache 19.04 and 19.91
  in two runs (+18% to +23%), cache off 14.86. Prefill unchanged.
- Why real text gains less: the server's 8K context left a smaller pool (1671 MiB, 397 slots), and real text hits far
  less: 45.7% against 76 to 82% for random tokens at similar sizes, with about 19 fills per generated token.
- Routing skew (imatrix, 61k tokens of code and docs): per layer, 80% of reads need 10 to 19 of 32 experts, and no
  expert is unused in any layer. The best static hot set at 1668 MiB would hit 53.7% (in-sample, an upper bound)
  against the cache's measured 45.7%.

What this means:
- The lossless cache pays on even a 35 W, PCIe 3.0, 4 GB card: about +20% decode on real text, no prefill cost.
- Measure with real text. llama-bench's random tokens overstated the gain threefold.
- A warm start or static hot set from imatrix is worth at most a few points of hit rate for gpt-oss-20b. It may matter
  more for models with more and finer experts (Qwen3.6-35B-A3B: 256 experts), which brick cannot hold in RAM at Q8.
- Pruning gpt-oss MAPLE-style would cut experts that code text uses; none is idle.

## Results on brick, morning of 2026-11-10: both techniques on one model

- The KV stream runs Qwen3.6-35B-A3B (UD-Q4_K_M, the file InsiderLLM measured) after three one-line architecture-gate
  changes on brick's fork (branch brick/qwen35moe, commits 3bc505a81 and a82d9782d). Probe decode 19.1 t/s with the
  stream at a 32K window against 19.5 stock at 8K; the needle was found in every run.
- Stock's window for this model on brick: 98,304 loads, 114,688 does not (whether stock serves a 97K prompt was not
  established; the run was stopped). The stream beyond stock's window was not measured.
- Upstream's `--moe-cache-mib` slows decode on brick: gpt-oss-20b 8.63 against 14.50 t/s; Qwen3.6 on the stream 9.7
  (400 MiB) and 12.8 (800 MiB) against 18.4. leloch's design gained on gpt-oss (19 to 20 against 16.2 stock best).
  So on a slow-PCIe, small-GPU machine the CPU-computes-misses design is the one that pays. Combining it with the
  stream needs a merge: 41 conflict hunks in 14 files, both forks share the base 15586e2.
- leloch's cache on Qwen3.6, real text, 8K, no stream: 19.84 t/s against 19.81 stock (no gain). The cache held
  4.4% of the expert tensors (256 small experts per layer, 8 used) and hit 32.4%; moving two layers' experts to the
  GPU gained nothing either. The cache pays for few large experts (gpt-oss: 32 per layer, 4.2 MiB per tensor), not
  for many small ones. Combined with the stream's break-even near 32K, there is no practical case for combining the
  two on brick.
- For beast: try `--moe-cache-mib` with Qwen3.6-35B-A3B on HIP first (already in the fork); PCIe 4.0 and a faster
  GPU may well change the verdict.

## Results on beast, 2026-11-10 late morning

RX 9070 16 GB, Ryzen 9 5900X, 60 GB RAM, PCIe 4.0. Upstream's cache (`--moe-cache-mib`, the SLRU version) in the
fork's HIP build (`build-hip`, 2026-10-29). The same Qwen3.6-35B-A3B UD-Q4_K_M file as on brick (SHA-256 ac0e2c11...)
and the same real-text prompts (brick's realtext.py and its four excerpts, copied). `-ngl 99 -fa on -c 8192 -t 12
--fit off`, q8_0 K and V. Scripts and logs: the session scratchpad, beast-cache/.

Stock sweep (llama-bench tg64): `-ncmoe` 24 / 20 / 18 / 16 / 15 = 40.0 / 46.2 / 49.5 / 52.9 / 53.1 t/s; 14 and below
do not fit in VRAM.

| config (real text, 3 requests of 400 tokens) | decode t/s | prompt reading t/s |
|---|---|---|
| stock `-ncmoe 16` | 52.79 | 950-1075 |
| `-ncmoe 16` + cache 1500 MiB | 54.64 (+3.5%) | 860-1050 |
| all experts in RAM + cache 10000 MiB | 55.31 (+5%) | 670-770 (-25 to -30%) |
| all in RAM + cache 6000 MiB | 46.71 | 490-610 |
| all in RAM, no cache | 31.57 | 490-570 |

- Upstream's cache helps on beast but only a little: +3.5 to +5% decode over the best stock placement, because stock
  already keeps 24 of the 40 layers' experts in VRAM. On brick (PCIe 3.0, small GPU) the same cache was slower at
  every size; on beast the fast bus makes its uploads cheap.
- Moving all experts to RAM and caching them buys the most decode but costs a quarter of prompt reading speed;
  `-ncmoe 16` plus a small cache keeps prompt reading and still gains 3.5%. Neither is a big enough win to change the
  everyday setup on its own.

Coding challenge on beast (the duration parser, brick section 18): Pi 1.5.1 with Qwen3.8 27B (router, thinking off),
"Do the task in TASK.md.": hidden harness 24/24, its own make test passes, 0 warnings, valgrind clean, 8 min 10 s
including the cold model load. brick (Qwen3.6-35B-A3B): the same 24/24 in about 11 minutes.

## Not checked yet

- Whether leloch's branch builds with CUDA 12.4 and gcc 15.2 (brick's toolchain).
- Whether gpt-oss-20b's graph hits the cache's fused gate/up path (gpt-oss uses a clamped SwiGLU with biases; the
  doc says other GLU graphs use the stock path, which should still cache the plain `MUL_MAT_ID` nodes).
- Real routing skew on any of our models; the script was only tested on synthetic counts.

### Qwen3.6 Q8_0 against Q4 on beast (2026-11-10, 14:12)

Same HIP build and real-text test as the Q4 table above. Q8_0 is 34.4 GiB; at the start 37 GB of RAM was available
(8.8 GB of /tmp, a RAM disk, held another session's files; swap full), so the file barely fits in the page cache.
A first attempt at 13:46 ran while Ollama held 7.4 GB and is discarded (0.4 to 12 t/s, pure paging).

- Sweep, llama-bench tg64: `-ncmoe` 22 does not fit; 24 / 26 / 28 / 30 / 32 / 34 = 16.0 / 2.6 / 22.3 / 14.1 / 2.0 / 11.9
  t/s. The scatter is paging, not the setting; 28 was best.
- Real text at `-ncmoe 28`: decode 27.4 t/s (24.3, 25.7, 28.0, then 28.6 on the warm-up request, still rising);
  prompt reading 27, 141, 263, 382 t/s, rising as the file came into the page cache.
- Q4 at `-ncmoe 16`: decode 52.8 t/s, prompt reading 950 to 1075 t/s.

So the Q4 decodes about twice as fast and reads prompts about three times as fast or more. The Q8 numbers are a lower
bound: with about 5 GB more free RAM it would not page, and its steady state would be somewhat higher.

Rerun at 14:36 with 52 GB of RAM available (David freed memory), no paging:

- Sweep, llama-bench tg64: `-ncmoe` 24 / 25 / 26 / 28 / 30 / 32 = 31.0 / 30.5 / 30.0 / 28.9 / 28.0 / 26.8 t/s;
  22 does not fit (23 not tried).
- Real text at `-ncmoe 24`: decode 31.0 t/s; prompt reading 515 to 590 t/s (the 189 t/s warm-up request excluded).

| Quant | Best `-ncmoe` | Decode t/s | Prompt reading t/s | RAM for the file |
|---|---|---|---|---|
| Q8_0 | 24 | 31.0 | 515 to 590 | 37 GB |
| UD-Q4_K_M | 16 | 52.8 | 950 to 1075 | 22 GB |

The Q4 decodes 1.7 times as fast and reads prompts 1.8 times as fast. The Q8 needs about 37 GB of free RAM to run
at this speed; with 37 GB free (the 14:12 run) it already paged and lost a little, and below that it collapses.

### Coding challenge on the Q4 (2026-11-10, 15:48)

Same duration-parser task and hidden 24-test harness as qwen38 above, through Pi (`beast-moe/qwen36`, a temporary
HIP server on port 8091: UD-Q4_K_M, 64K context, `-ncmoe 16`, 15.1 GB VRAM). Result: 24/24, zero warnings, valgrind
clean, the agent's own 28 tests pass. Time 2 min 34 s, against 8 min 10 s for qwen38. brick's run (its DOCUMENTATION.md, section 18) was this same
Q4 file: also 24/24, in about 11 min, after it found and fixed a skipped-unit bug through its own tests.

### Q4 context size, and an unexplained slowdown (2026-11-10, afternoon)

Fit test (load plus one request, 16304 MiB VRAM): 128K fits at `-ncmoe` 16 (177 MiB spare) and 17 (about 900 MiB
spare); 256K needs 18 (22 MiB spare), 19 (319) or 20 (784). Chosen for the router preset: 128K at `-ncmoe 17`
(David: 256K is more than small tasks need). Real text at 128K/17: 41.0 t/s decode, 910 to 970 t/s prompt reading;
the same test at 8K/16 gave 41.3, so 17 costs nothing. KV at q8 is about 11 KB per token (10 of 40 layers attend).

Both are about 22% below the morning's 52.8 t/s with the same flags and file. Not CPU threads (8 to 12 all about 45
t/s in llama-bench), RAM (54 GB available), temperature or Ollama. David rebooted beast before rechecking. Scripts and
logs from today: ~/ml/bench/llama-2026-11-10/ (moved there from the RAM disk /tmp; the challenge runs are in its challenge-runs/).

### Router presets: Vulkan build against the pinned HIP build (2026-11-10, after the reboot)

beast rebooted (kernel 7.0.0-43); the Q4 reference test was back to 50.8 t/s at 8K/`-ncmoe 16` (morning 52.8), so
the afternoon's 41 t/s was a machine state, not the setup. Pinned serving build: a git worktree of the fork at
17692e571, `~/store/code/katalystnord/llama.cpp-kvstream-serve/build-hip` (build 11899). Same real-text test, 16K
context, the exact flags of each router preset (`~/ml/bench/llama-2026-11-10/beast-cache/compare.sh`):

| Preset | Decode Vulkan | Decode HIP | Prompt reading Vulkan | Prompt reading HIP |
|---|---|---|---|---|
| qwen38 (MTP + n-gram) | 47.4 | 66.6 (+41%) | 735 to 775 | 715 to 720 (-4%) |
| saluki (n-gram) | 37.1 | 37.6 (same) | 860 to 910 | 600 to 610 (-30%) |
| gpt-oss (n-gram) | 145.9 | 120.7 (-17%) | 3770 to 4260 | 2350 to 2400 (-40%) |
| Qwen3.6 Q4 (`-ncmoe 16`) | about 31 (Oct) | 50.8 (+65%) | | 1030 to 1090 |

MTP acceptance was the same on both (qwen38 0.51 and 0.54), so the qwen38 gain is the backend, not the drafting.
The router starts every child with its own binary (server-models.cpp, `get_server_exec_path()`); a preset cannot pick
a different one.

gpt-oss accuracy is not a backend matter: perplexity on the same 320 KB of text (`-c 1024`) was 3.0719 on Vulkan and
3.0744 on HIP (each +/- 0.034). Pi sends gpt-oss no `reasoning_effort` when its thinking level is off, so the
template uses "medium"; `/thinking high` sends "high".

Dual build (HIP and Vulkan in one binary, `llama.cpp-kvstream-serve/build-dual`, 5 min to build): it lists the card as
ROCm0 and Vulkan0, and `--device` puts all weights on the named one (verbose load: gpt-oss 8 Vulkan0 buffers and no
ROCm0; Qwen3.6 Q4 6 ROCm0 buffers and no Vulkan0). Each preset on its best backend, same real-text test:

| Preset | Device | Decode t/s | Prompt reading t/s |
|---|---|---|---|
| qwen38 | ROCm0 | 65.8 | 720 to 725 |
| saluki | Vulkan0 | 40.9 | 800 to 870 |
| gpt-oss | Vulkan0 | 144.2 | 4180 to 4550 |
| Qwen3.6 Q4, 128K, `-ncmoe 17` | ROCm0 | 50.3 | 1000 to 1060 |

Every preset matches or beats the better of the two single builds above.

Switched 2026-11-10 evening. The serving worktree lives on the ML SSD, `~/ml/llama.cpp-kvstream-serve` (David: never
serve from a code directory): `build-dual` for the router, `build-hip` for llama-long.service. All six presets load
through the router on their device; qwen36-q4 runs its 0.9 GB projector on the CPU (`mmproj-offload = false`; on
the GPU it took VRAM to 16295 of 16304 MiB), 15339 MiB in use, a 1280x800 image in 8.6 s. The duration challenge
through the router (`beast-llamacpp/qwen36-q4`): 24/24, 54 of its own tests, valgrind clean, 3 min 3 s.

## Nemotron 3.5 Bolt 74B-A9B 24GB on beast (2026-11-10)

2-bit QAT file, 22.0 GB, 88 blocks (Mamba-2, attention and MoE mixed), 64K context, q8_0 KV, `build-dual`.
Logs and scripts: `~/ml/bench/llama-2026-11-10/beast-cache/` (`bolt.sh`, `bolt-fit2.sh`, `bolt-fit*.txt`, `bolt-speed.txt`).

Fit: ROCm0 fails with out of memory below ncmoe 40, is full at 40 (16277 MiB) and fits from 44 (15394 MiB), 48 (14919
MiB). Vulkan0 reports "fits" even at ncmoe 0 with VRAM at the ceiling, so the first sweep's Vulkan results mean
nothing; the second sweep logged GTT and showed a steady +700 MiB host buffer at every setting.

| Device, ncmoe | Plain decode | DFlash (LiLiCorr, 6 draft) |
|---|---|---|
| ROCm0, 48 | 31.5 t/s (prompt 593 t/s) | drafter out of memory |
| ROCm0, 56 | 27.6 t/s | 29.4 t/s, acceptance 0.38 |
| Vulkan0, 48 | 26.1 t/s | 14.4 t/s, acceptance 0.36 to 0.54 |

Preset: ROCm0, ncmoe 48, no drafter. Decode matches the Qwen3.6 Q8 (31.0) and is well below the Q4 (50.3).

## MTP crash with draft-mtp,ngram-mod (fixed 2026-11-10, aa7aa45f7)

qwen38 through Pi aborted on `GGML_ASSERT(i_h < n_verify_h_max)` in `common/speculative.cpp` while writing C code (once
in the first duration run, 6 times in the second). The MTP state kept a fixed 64 verify rows; ngram-mod drafts up to
64, so a verify has 65 rows and a fully accepted draft asked for row 64. The cap is now 1 + the longest draft in the
chain. Repro (`~/ml/bench/llama-2026-11-10/mtp-fix/repro.sh`, copy a 76-line C file back verbatim): old build aborts
in round 1, fixed build runs 3 rounds at mean draft length 60.9, acceptance 0.94, about 290 t/s. Real-text decode
unchanged (65.6 t/s against 65.8). Serving worktree moved to aa7aa45f7, both builds rebuilt.
