# Roadmap: KV streaming as the everyday server

Written 2026-10-30. Goal: run Qwen3.8 27B with the KV stream as the default, everyday server on the RX 9070 (non-XT,
16 GB), at a 64K window or more with q8_0 K and V, instead of a stock server capped at 48K. Today the stream runs as a
separate long-context server that is switched on when needed.

## Why it is not the default yet

1. **The copy stall** ([design/copy-stall.md](design/copy-stall.md)): in streaming mode a deep read can deadlock
   (7 of 11 deep reads of Saluki 27B at a 256K window; Qwen3.8 passed its 3 runs at 250K, but the mechanism is the
   same). For the server that editors, agents and chat all lean on, a rare hang is worse than a smaller window.
2. **One model, one shape**: the stream server runs one model at a time, and only models with head size 256 and 6
   query heads per KV head (the paged kernels' shape). A stock router serves several models on demand.
3. **An older base**: the fork is based on llama.cpp of 2026-10-21 (a37cc4610) and lacks later upstream work, such as
   unloading after idle time.

Not a reason: speed. The README's comparison (same probe, HIP build) has the stream faster than plain at short context
(55.1 against 43.6 t/s at 8K) and nearly flat to 64K, but the everyday server is a different build (Vulkan), so the
two have not been measured like for like.

## Steps, in order

### 1. Measure like for like (short; do first)
- The same probe (`bench/longctx.py`) on the everyday server and on the stream server at 8K, 32K and 64K: write and
  read speed, VRAM, host RAM.
- The coding harness on the stream server at 64K with q8_0 K and V. The everyday server at 48K scored 22 of 22; the
  stream server must match it before it replaces it. (A q4_0 V cache made the model loop in the same harness.)
- Outcome: how much the stream gains in daily use. If it is not faster, the case rests on the larger window and on
  read once, ask later, which is still a case for coding, but should be known before the bigger work.

### 2. Fix the copy stall (the blocker; size unknown)
- Choose one of the three options in [design/copy-stall.md](design/copy-stall.md): do the evict and assign copies
  with kernels so HIP issues no SDMA copies with dependencies for them; give the copy engine its own SDMA queue through
  libhsakmt; or order HIP's copies after the copier has queued the ubatch's layers. Ideally agreed with the copy
  engine's author first (report sent 2026-10-29 on achyudh/llama.cpp pull request 1).
- Done when: `bench/repro-stall.sh` gives about 20 clean deep reads in a row, on both Qwen3.8 and Saluki (the bug hit
  7 of 11), with no change in output (verify mode) and no loss of speed beyond noise.

### 3. Bring the fork up to date
- Rebase onto current llama.cpp; rerun the stall reproduction, the probe and the coding harness afterwards.
- Decide the long-term home: keeping a fork current is work that never ends. Prefer getting the stream, the split and
  the stall fix upstream (into the copy engine's branch, or llama.cpp), so the fork can shrink.

### 4. Wire it in as the default
- Route requests for Qwen3.8 to the stream server and everything else to the stock router, behind the same address,
  so editors, agents and chat clients see no change.
- The two servers must not compete for the GPU: today switching to the stream server pauses the router.
- Expose read once, ask later where it pays: agents first (a long document read once, then questions in seconds).

### 5. Make it robust
- A stall that slips through must end the request cleanly (the stream's 30 s watchdog) and restart the service, not
  hang it; clients retry.
- Health checks and monitoring like the other services; a note of the memory budget (host RAM for the KV cache at the
  chosen window).

## Rough effort
Steps 1, 3, 4 and 5: about a day each. Step 2: days, depending on the option. Order: 1, 2, 3, 4, 5; step 1 can run
while the stall fix is being discussed.

## Side tracks (not on the KV-stream path, kept so they are not lost)

### Nemotron 3.5 Bolt 74B-A9B: faster on beast (added 2026-11-10)
Bolt scored full marks on the three agent tasks (52/52, `~/ml/bench/agent-tasks-2026-11/README.md`), as good as
Qwen3.8 but about 3x slower end to end: 31.5 t/s decode on ROCm0 at 64K with ncmoe 48, against 65.6 for Qwen3.8.
Measurements and fit sweep: [design/expert-tiering.md](design/expert-tiering.md), section "Nemotron 3.5 Bolt".
Facts from the file: 88 blocks, 496 experts, no MTP head; experts are NVFP4 (80 tensors, 15.9 GB of 22 GB), the rest
IQ2_XXS and Q8_0. NVIDIA's DFlash drafter (LiLiCorr) does not pay off here (29.4 against 27.6 t/s at ncmoe 56, halves
speed on Vulkan).

In order:
1. **Profile the NVFP4 expert path** (about 15 minutes on the card, no download). RDNA4 and the 5900X have no
   native FP4, so NVFP4 is dequantized in software on both; at ncmoe 48 more than half the experts run on the CPU.
   Find out whether the NVFP4 matmul kernels (HIP and CPU) are the bottleneck. If so, this is kernel work in the fork
   and likely the biggest gain.
2. **Expert hot-set instead of whole layers**: with 496 experts, keeping only the most used ones on the card
   (`kvstream/scripts/expert-hotset.py`, the expert-tiering work) may beat `-ncmoe`, which moves whole layers.
3. **Requantize the experts** NVFP4 to Q4_K or IQ4 for the tuned kernels. A small loss from 4-bit to 4-bit; must be
   rescored on the agent tasks.
4. **Smaller community 2-bit files** (ask David before downloading): AtomicChat AD-IQ2_XS, lmstudio-community
   IQ2_XXS. Smaller means more of the model on the card. Sizes and quality not checked.
5. **A Saluki-style build** (large: BF16 download well over 100 GB, days of work): mixed precision per layer with an
   importance matrix from our own agent sessions, tuned to keep tool calling. Only if 1 to 4 leave Bolt worth it.
