# The copy stall: a deadlock between the stream's copies and HIP's write-backs (2026-10-29)

## Summary

While a long prompt is read with `--kv-stream` (Saluki 27B at a 256K window on the RX 9070), the server sometimes stops:
one layer's page copy never completes, and the stream's 30 s watchdog aborts (or, with a longer watchdog, the
attention kernel's 60 s trap fires). The cause is a deadlock between two users of the same ROCm SDMA queue:

1. The stream's copy engine (`ggml/src/ggml-cuda/copy-engine.cu`) submits its host-to-device page copies with
   `hsa_amd_memory_async_copy_on_engine` on engine id 0x2. ROCr turns the engine id into a blit queue index with
   `ffs()`, so 0x2 selects index 2, `BlitDevToHost`: ROCr's device-to-host queue, not a private queue on a spare
   engine.
2. HIP writes evicted pool pages back to host memory (`ggml_backend_tensor_get_async`, queued on the compute stream).
   It picks an SDMA engine per copy; when it finds no engine free, it falls back to the plain
   `hsa_amd_memory_async_copy`, which picks the queue by direction: device-to-host, the same `BlitDevToHost` queue.
3. The write-back depends on earlier work on the compute stream (`wait_event`). ROCr implements a dependency as a
   `POLL_REGMEM` packet inside the SDMA queue, so the queue stops there until the dependency is met.
4. That earlier work includes the attention kernels of the ubatch in flight, which wait (`ggml_flash_attn_ext_set_sync`)
   for page copies the copier queues later, behind the poll. Nothing can move.

It depends on timing: the fallback happens only when both SDMA queues have pending bytes at the moment of a
write-back, and the copier must still have layers of the current ubatch to queue. Deeper prompts make the stream's
queue busier, so the stall appears in deep runs. It is not a size limit or a hardware fault.

The copy engine is achyudh's (`feature/adaptive-kv-stream-rocm`); this repository has not changed it. The split
attention work is not involved (the stalls happen with `LLAMA_KV_SPLIT` unset, and first happened before it existed).

## Symptoms and frequency

- Saluki 27B (IQ2), q8_0 K and V, decode pool 2420 or 5400 MiB, 256K window, reading a 250K prompt, no drafting:
  7 of 11 reads stalled (2026-10-28 and 2026-10-29), 6 of them at the ubatch near 166K tokens (tag 298), the one
  with HIP's logging on at about 90K (tag 149). Qwen3.8 at 250K passed in its 3 runs.
- The stream's own state at the stall is always "jobs N+1 queued / N-1 done, layer copies queued = published + 2": one
  layer copy submitted and never done, the next one queued behind it.
- Server log (traced run, 21:55):

      E wait_for: stalled for 30.0 s waiting for copies to land (layer 9, tag 298)
      E log_state: tag 299, jobs 299 queued / 297 done, layer copies 4763 queued / 4761 published

- With `LLAMA_KV_STREAM_TIMEOUT=900` the GPU trap fires first, after 60 s: `HSA_STATUS_ERROR_EXCEPTION ... code:
  0x1016` (the deliberate `__builtin_trap` in `pd_sync::wait`). So the copy did not arrive within 60 s, where it
  normally takes about 15 ms: it is lost, not slow.

## Evidence

Reproduction: `kvstream/bench/repro-stall.sh` (Saluki, 256K window, 2420 MiB pool, up to three reads of the 250K
probe, stops at the first stall; `TIMEOUT` sets the watchdog). Copy engine on SDMA 0x2 of 0x3 available (its log line).

Kernel log: no amdgpu faults, SDMA timeouts or GPU resets at any stall.

bpftrace (uprobes on `hsa_amd_memory_async_copy_on_engine` in libhsa-runtime64 1.18.0, kprobes on the KFD eviction,
restore, quiesce, userptr and HMM paths), traced run, seconds after the trace start:

    469.6913  main thread: 64 copies of 278,528 bytes, device -> host (two evicted pages, K and V, 16 layers)
    469.7248  copier: layer 9 of tag 298, 2 x 166,002,688 bytes host -> device (never completes)
    469.8424  copier: layer 10 of tag 298, 2 x 166,002,688 bytes

- Every copy call returned success. Copy sizes grow smoothly (2 pages per ubatch).
- KFD queue evictions came in matched evict/restore pairs early in the run (model load; memory compaction by
  kcompactd0 at 64 s and 130 s); none between 131 s and the abort; the one unmatched eviction is the abort itself.
  So the driver did not suspend the queue.
- Write-back bursts before a copier copy are routine (262 in the run, many closer than this one), so the order alone
  is not the cause: what matters is the queue the write-back lands in.

HIP's copy log (`AMD_LOG_LEVEL=4 AMD_LOG_MASK=768`, decimal: HIP did not take 0x300), logged run, the last HIP
copies before the stall:

    HSA Copy copy_engine=0x1, dst=0x755f5c36c000, src=0x75635911c000, size=278528, ..., wait_event=0x756491e38a00, ...
    Query copy engine status 1008, ..., free_engine_mask 0x0, rec_engine_mask 0x0
    HSA Copy dst=0x755f6d36c000, src=0x756359eec000, size=278528, wait_event=0x756491e38980, ..., engineType=1
    (two more pairs like the last two)

- Over the run HIP chose engine 0x1 for all 11,241 of its engine-picked copies; the last three copies, the write-backs
  right before the stall, found no engine free (status 1008 = out of resources) and went through the plain API,
  each with a `wait_event`.

## The code path (Ubuntu's ROCm 7.1 packages: rocr-runtime 7.1.0+dfsg-0ubuntu9, rocm-hipamd 7.1.0-0ubuntu2)

ROCr (`runtime/hsa-runtime/core/runtime/`):
- `runtime.cpp`, `Runtime::CopyMemoryOnEngine`: `int engine_offset = ffs(engine_id);`
- `amd_gpu_agent.h`: `enum BlitEnum { BlitDevToDev, BlitHostToDev, BlitDevToHost, DefaultBlitCount };`
  so engine 0x1 -> `BlitHostToDev`, engine 0x2 -> `BlitDevToHost`.
- `amd_gpu_agent.cpp`, `GpuAgent::DmaCopyOnEngine` -> `GetBlitObject(engine_offset)` -> `blits_[engine_offset]`.
- `amd_gpu_agent.cpp`, `GetBlitObject(dst_agent, src_agent, size)` (the plain `hsa_amd_memory_async_copy`):
  device-to-host -> `GetBlitObject(BlitDevToHost)`.
- `amd_gpu_agent.cpp`, blit queue creation: `rec_eng = -1` when the device has no xGMI SDMA engines, so on this card
  the queues are not pinned to engines; the engine ids name queues.
- `amd_gpu_agent.cpp`, `DmaEngineIsFree`: a queue is free if never used or with no pending bytes.
- `amd_blit_sdma.cpp`, `SubmitLinearCopyCommand`: each non-zero dependency signal becomes `SDMA_PKT_POLL_REGMEM`
  packets ahead of the copy in the same queue.

HIP (clr, `rocclr/device/rocm/rocblit.cpp`, `DmaBlitManager::hsaCopy`):
- engine: the last used, else `memory_copy_engine_status` + `memory_get_preferred_copy_engine` (the recommendation
  is empty except on gfx94x); lowest free engine;
- if that query fails (no engine free): `kUseRegularCopyApi = true` -> `Hsa::memory_async_copy(...)` with the stream's
  wait events.

## Fix options (not yet chosen)

Moving the stream's copies to engine 0x1 does not help: HIP uses that queue for most of its own copies, with the same
dependency polls. Through `hsa_amd_memory_async_copy[_on_engine]` every SDMA queue on this card can receive HIP's
polls. Options:

1. No HIP SDMA copies with dependencies while the stream copies: do the stream's write-backs (evict) and page loads
   (assign) with a small copy kernel on the compute stream (pinned host memory is device-visible), so HIP issues no
   SDMA copies for them. Other HIP device-to-host copies (logits read-back) need checking too.
2. A queue HIP cannot reach: create the copy engine's own SDMA queue through libhsakmt (`hsaKmtCreateQueue` with an
   SDMA queue type) and write its own copy packets. Robust, more code.
3. Ordering: before a HIP copy that can depend on the ubatch in flight, wait until the copier has queued all of that
   ubatch's layer copies, so nothing the dependency needs can sit behind it. Simplest, but it must cover every HIP
   copy (write-backs, read-backs), and it costs host/GPU overlap.

To discuss with achyudh before choosing; then verify with `repro-stall.sh` (7 of 11 deep reads stalled, so a fix needs
several clean runs to count).

## Draft report for achyudh (not sent)

> Title: kv-stream: deadlock between the copy engine's SDMA copies and HIP device-to-host copies (gfx1201, ROCm 7.1)
>
> With --kv-stream on an RX 9070 (ROCm 7.1, Ubuntu packages), reading a 250K prompt at a 256K window intermittently
> stalls: one layer's copy never completes and the watchdog aborts (7 of 11 reads, Saluki 27B, 2420/5400 MiB pool).
> The copy engine submits on engine id 0x2, which ROCr maps with ffs() to its BlitDevToHost queue. When HIP writes an
> evicted page back (tensor_get_async on the compute stream) and finds no SDMA engine free, it falls back to
> hsa_amd_memory_async_copy, which puts the device-to-host copy on the same BlitDevToHost queue, with a POLL_REGMEM on
> the stream's previous work. The attention kernels of the ubatch in flight wait for copies queued behind that poll:
> deadlock. Evidence: HIP's copy log (status 1008, free_engine_mask 0x0, plain-API write-backs right before the
> stall), a bpftrace of the submissions (no driver evictions, no errors), and the ROCr/clr source paths. Reproduction
> script and details: <link to kvstream/design/copy-stall.md>. Happy to test a fix or send one.

## Investigation material removed (2026-10-29)

Removed after this write-up, as agreed with David: the umr build (~/code/scratch/umr), the ROCr and HIP sources
(~/code/scratch/rocr-src), the bpftrace script and trace and HIP's copy log (~/code/scratch/stall-trace), and the
reproduction server logs. The excerpts above are what was kept. `kvstream/bench/repro-stall.sh` stays, to verify a fix.
