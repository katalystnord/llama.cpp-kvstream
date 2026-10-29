# Design note: a tiered KV cache scheduler (idea, 2026-10-28)

Status: idea, not started. Come back to it after the q8 V question and the performance sweep (see `../README.md`).
Origin: David Sandquist, after the streaming and save/restore tests: "a scheduler to say what goes where on the streaming memory".

## What we have now

- The fork keeps the KV cache in host RAM and a fixed pool of pages on the GPU (`--kv-resident-mib` 450 MB for
  prefill, `--kv-resident-mib-decode` 2420 MB for decode), with a page table, a copy engine beside the compute stream,
  and a prefetch of the next decode step's first layers.
- Slot save/restore works with streaming: a 125K-token document saves to a 4.2 GB file in 25 s and restores on a cold
  server in 2.7 s; the next question reads only its own tokens.
- the test machine's link is PCIe 4.0 x16 (Ryzen 9 5900X root port at 16 GT/s), about 32 GB/s; the fork's author had PCIe 5.0.
- Qwen3.8 is hybrid: only 16 of 64 layers keep a growing KV cache (4 KV heads x 256); the delta-net layers keep a
  fixed-size state. KV is about 34 KB per token in the slot file (q8 K / q4 V plus states).

## Three levels where a scheduler can help

1. Within one forward step: ordering, not choice. Full attention reads every page every token, so placement cannot
   avoid the traffic; a scheduler can only hide it by fetching layer N+1's pages while layer N computes. Measure
   first how much PCIe time is already hidden (copy-engine vs compute timelines at 64K, 128K, 195K), and whether a
   larger resident pool or a different split between prefill and decode pools helps.
2. Within one step: choice (approximate). Query-aware page selection (Quest-style per-page min/max key summaries,
   InfiniGen-style speculative prefetch, ECHO for sparse-attention models; LiteCache: neighbouring queries reuse most
   of the same top-k pages) fetches only the pages that matter for this token. Big win on a PCIe-limited machine, but
   it changes the model's output, so every change must pass the code-word recall test at depth and the summary check.
   No llama.cpp implementation exists (2026-10 search); vLLM Ascend has a version for DeepSeek-style sparse models.
3. Across documents and sessions: tiers. VRAM (hot), host RAM (warm, 60 GB), disk (cold slot files on the 15 TB data
   disk). A scheduler promotes and demotes whole documents (or page ranges) by use, like an OS pages memory:
   - documents read overnight become slot files; a question about one loads it in seconds (2.7 s from disk today,
     less from RAM);
   - several warm documents can sit in RAM at once (6 GB per 200K-token document), so switching between them avoids
     the disk;
   - eviction by recency and size; pinning for documents in active use.
   This is the most practical level for document-heavy work (a library of pre-read papers, patents, regulations).

## Host memory placement: measure the pipe first (David Sandquist, 2026-10-28: "like BadRAM, measure where to allocate")

Checked on the test machine: one NUMA node (Ryzen 9 5900X, one memory controller on the I/O die), dual-channel interleaving, so no
physical region of RAM is faster than another. What can differ is how a page reaches the GPU:
- already pinned: the stream's host pages come from the backend's host buffer type (`hipHostMalloc`);
- already untranslated: the GPU's IOMMU group is `identity` (pass-through), so `iommu=pt` would change nothing;
- open: 4 KB vs 2 MB (huge page) backing of the pinned pool; coherent (`hipHostMallocCoherent`, used by the copy
  engine in `ggml/src/ggml-cuda/copy-engine.cu`) vs non-coherent host memory; transfer chunk size; overlap with
  compute. PCIe 4.0 x16 tops out near 25 GB/s in practice; dual-channel DDR4 gives about 45-50 GB/s, so PCIe is the
  limit.
First tool: a calibration microbenchmark (host-to-device bandwidth for each allocation kind and chunk size, with and
without concurrent kernels), then allocate the stream pool the fastest measured way.

## User side: a context lever in Pi (David Sandquist, 2026-10-28)

When loading a model in Pi, set the context window with a slider and see what fits: a bar splitting VRAM into weights,
MTP head, resident KV and working buffers, and the part of the KV cache that streams from host RAM, with the expected
write speed at the chosen depth (from measured tables). Confirming restarts the fork server with that window and V type
(`kvstream/scripts/serve-long.sh`) and switches Pi to a model entry with the matching contextWindow. Parts: a memory model per
configuration (KV about 35 KB/token with q8 V, 27 KB with q4 V for Qwen3.8's 16 attention layers; calibrate), a small
control script, a Pi extension (same machinery as the mode switcher and footer in ~/.pi/agent/extensions/modes.ts),
and short calibration runs for the speed estimates.

## Where it should live

Inside llama.cpp (this fork), not in the Linux kernel: the manager needs to know the model's layers, which layers
have KV, the attention pattern and the page table, all of which live in `src/llama-kv-cache.cpp`, the streaming code
(`llama_kv_stream`) and `ggml/src/ggml-cuda/copy-engine.cu`. Level 3 could start outside the server, as a small
service that keeps a catalogue of slot files and calls the server's save/restore API.

## First steps when we pick this up

- Level 3 prototype: a catalogue (document id, token count, slot file, last used) and a client that restores before
  asking; measure restore from page cache (RAM) vs cold disk.
- Level 1 measurement: profile copy-engine and compute overlap per layer at 64K/128K/195K; try resident pool sizes.
- Level 2 only after that, with the recall tests as the gate.
