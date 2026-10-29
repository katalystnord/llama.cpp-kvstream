# kvstream: measurements, scripts and design notes

Everything Katalyst Nord adds to the fork outside the C++ and GPU code lives in this folder. The code changes
themselves are ordinary commits in the tree (see the root [README](../README.md) for the list).

| folder | contents |
|---|---|
| `scripts/` | `build-rocm.sh` (HIP build for gfx1201), `serve-long.sh` (long-context server) |
| `bench/` | the long-context probe (`longctx.py`), save/restore test, and the run scripts behind every table below |
| `results/rx9070-ryzen5900x/` | the raw logs of those runs |
| `design/` | `split-attention.md` (GPU and CPU attending over one cache in parallel), `tiered-kv-scheduler.md` |
| `integrations/` | a context lever for the Pi coding agent, and the systemd unit used to run the server |

## Test machine

AMD Radeon RX 9070 (the non-XT card: 56 compute units, 16 GB, gfx1201), AMD Ryzen 9 5900X, 60 GB dual-channel DDR4,
PCIe 4.0 x16, Ubuntu with Ubuntu's ROCm 7.1 packages. GPU power cap 198 W or 220 W as noted per table.

The fork's own numbers (see [README-kvstream-fork.md](../README-kvstream-fork.md)) are from an RX 9070 XT with
a different quantisation (IQ4_XS) and recommend PCIe 5.0 and DDR5. The XT has 64 compute units and a higher power
limit, so its numbers and ours are not directly comparable; ours are what a cheaper card on an older platform gets.

## The probe

`bench/longctx.py <port> <tokens>` builds a prompt of the requested length from llama.cpp's own source files (public
text, so anyone can rebuild it), hides one sentence at 10% depth, and at the end asks for the code word in that
sentence and a short summary of one of the files. The sentence is
"the code word for the beast test is MARMALADE-7"; "beast" is the test machine's name, kept so that prompts and token
counts stay identical to the logged runs. A run passes when the code word comes back exactly and the summary is
correct. It prints reading (prefill) and writing (decode) speed as reported by the server.

## Results

### Qwen3.8 27B on a 16 GB card (2026-10-28, 198 W)

Qwen3.8 27B IQ3_S, MTP + n-gram drafting with ReplaySSM (the fork's defaults).

| config | context | prompt tokens | read t/s | write t/s | code word | VRAM | host RAM |
|---|---|---|---|---|---|---|---|
| no streaming, q8 K / q8 V | 48K | 40,069 | 615 | 38.9 | found | 15.7 GB | 1.1 GB |
| streaming, q8 K / q4 V | 132K | 64,069 | 643 | 56.8 | found | 15.6 GB | 4.6 GB |
| streaming, q8 K / q4 V | 132K | 125,066 | 574 | 50.1 | found | 15.5 GB | 4.6 GB |
| streaming, q8 K / q4 V | 200K | 195,068 | 497 | 25.3 | found | 15.6 GB | 6.4 GB |

Without streaming the largest context that fits is about 48K. The 195K turn took 404 s, almost all of it reading;
follow-up questions on the same text reuse the cache. The summaries were correct at every depth.

### Q8_0 V cache in the paged decode kernel (2026-10-28, 220 W)

The fork's paged decode kernel took a Q4_0 V cache only. Q4_0 V made Qwen3.8 loop in our coding tests, so we added a
Q8_0 V variant (commit "kv-stream: Q8_0 V cache for the paged decode kernel"). Prompt reading routes Q8_0 V to the
stock kernels. `test-backend-ops`: 4448/4448 FLASH_ATTN_EXT cases pass against the CPU reference, including 178 paged
and 44 contiguous Q8_0 V cases.

| depth | Q4_0 V read / write t/s (198 W) | Q8_0 V read / write t/s (220 W) | code word |
|---|---|---|---|
| 64K | 643 / 56.8 | 577 / 52.5 | found |
| 125K | 574 / 50.1 | 464 / 34.0 | found |
| 195K | 497 / 25.3 | 379 / 16.4 | found |

Q8_0 V moves about 30% more data per token over PCIe. The reading gap also comes from the stock prefill kernels: the
fork's RDNA4 prompt kernel is Q4_0-only, and porting it is open.

With Q8_0 V, the Pi coding agent on Qwen3.8 at 96K context wrote at 57 to 64 t/s and read at 664 t/s, did not loop,
and finished a C library task with its tests passing (22/22 on a hidden test harness, 0 valgrind errors). The Q4_0 V
run of the same task looped.

### Read once, ask later (2026-10-28, 220 W, `bench/run-save-restore.sh`)

A 125,023-token document on the streaming server with `--slot-save-path`:

| step | time |
|---|---|
| read the document once | 212 s (590 t/s) |
| save the slot to disk | 25.5 s, 4.23 GB file (about 34 KB per token) |
| question 1, same server | 4.8 s total; only the 27 question tokens were read |
| cold restart of the server | 9 s |
| restore the slot from disk | 2.7 s |
| question 2 (the code word) | 1.9 s total: 34 tokens read in 0.85 s, answer at 57 t/s, code word correct |

A document read once can be questioned again within seconds instead of minutes. The restart bug reported upstream for
hybrid models (restored state dropped on the first query after a restart) did not occur here.

### Saluki 27B with vision, up to 250K (2026-10-28, 220 W, `bench/run-saluki.sh`)

Underdog Saluki 27B is a 2-bit (IQ2 mix) Qwen3.8 with the same attention shape (16 attention layers of 64, 4 KV heads
of 256, 6 query heads per KV head), so it streams with the same kernels and the same 34,816 bytes per token at q8 V.
It has no MTP head (n-gram drafting only), and its weights plus vision projector take 8.5 GB instead of 12.1, which
leaves room for a bigger GPU pool. After the long prompt, one scanned journal page goes through the projector.

| decode pool (`--kv-resident-mib-decode`) | depth | read t/s | write t/s | code word | VRAM peak |
|---|---|---|---|---|---|
| 2420 MiB | 64K | 497 | 33.8 | found | 12.2 GB |
| 2420 MiB | 125K | 423 | 16.2 | found | 12.6 GB |
| 5400 MiB | 125K | 424 | 30.2 | found | 14.6 GB |
| 5400 MiB | 175K | 367 | 25.9 | found | 15.2 GB |
| 2420 MiB | 250K (256K window) | 311 | 3.9 | found | 12.7 GB |

With 5400 MiB the whole 125K cache stays on the card while writing (up to about 156K at q8 V), so writing nearly
doubles; reading is unchanged because it uses the separate 450 MiB prefill pool. The page image was read correctly in
every run (table caption and all seven columns), also after the 175K and 250K prompts. The 250K turn took 13.9 minutes
of reading; writing at that depth streams about 6 GB of cache per token from RAM, so a full window suits reading a large
stack once and asking about it.

One run (pool 5400, 256K window, 250K prompt) stopped at 166K tokens of reading: a layer copy was queued but never
landed, and the stream's 30 s watchdog aborted (`results/.../saluki-stream-256k-pool5400-crash-server-2026-10-28.log`).
The same settings passed 175K on the next run, and the 2420 MiB pool passed 250K. Intermittent; cause open.

### Against the plain way (2026-10-28, `bench/run-nkvo.sh`)

Is the stream more than letting the cache spill into RAM? Same build, model (Saluki), drafting and probe, with the
cache in host RAM and attention on the CPU (`--no-kv-offload`) instead of `--kv-stream`:

| setup | 64K read / write t/s | 125K read / write t/s | VRAM |
|---|---|---|---|
| `--no-kv-offload` | 201 / 6.4 | 176 / 5.4 | 9.3 GB |
| `--kv-stream`, 2420 MiB pool | 497 / 33.8 | 423 / 16.2 | 12.6 GB |
| `--kv-stream`, 5400 MiB pool | | 424 / 30.2 | 14.6 GB |

Code word found everywhere. At 125K the stream reads 2.4x faster (5 instead of 12 minutes) and writes 3 to 5.6x
faster. The plain run's 5.4 t/s at 125K was helped by n-gram drafting (41 of 44 drafts accepted).

### Split attention, first pieces (2026-10-28, `bench/run-contention.sh`)

At depth, writing is bound by PCIe (about 24 GB/s measured). The CPU sits next to the host copy of the cache and is
idle meanwhile. Attention over disjoint sets of pages merges exactly through the log-sum-exp (LSE) of each part, so
the GPU can attend over the pages it has while the CPU attends over the rest in RAM. Design and arithmetic:
[design/split-attention.md](design/split-attention.md). Built and tested so far:

- CPU flash attention can write the per-row LSE (`tests/test-fa-lse.cpp` checks GPU part + CPU part = full attention).
- The paged GPU decode kernel writes the LSE too.
- A grouped-query CPU decode path that reads each K/V row once for all query heads sharing it: 14.4 to 21.6 GB/s
  of cache processed on the 5900X.
- Contention measured with the stream running (Saluki at 150K): stream 8.9 to 7.8 t/s, CPU attention 29 to 40-42 ms
  per token, both routes together about 35 GB/s of dual-channel DDR4. Expected gain from the split about 1.45x.

Not built yet: the CPU worker inside the stream that hands its share back for the merge. Until it is built and
measured, the 1.45x is an estimate. A blocked variant of the CPU kernel is on the branch `cpu-fa-block` (correct to
5e-8 on the CPU tests; GPU suite and speed not yet run).

## Build

`scripts/build-rocm.sh` builds `llama-server` and `llama-bench` into `build-hip/` with HIP through
`/usr/lib/llvm-21/bin/clang` (Ubuntu's ROCm 7.1 packages) and the fork's own cmake options. Two changes to the
fork's build, both in the server's web-UI step and none in the GPU code:

- `-DLLAMA_USE_PREBUILT_UI=OFF`: the fork's build number has no prebuilt UI on Hugging Face, so the download fails.
- `-DLLAMA_UI_GZIP=OFF`: gzipping the UI files failed across a symlinked source path.

With the prebuilt UI off, the script copies the UI files from another llama.cpp build (`UI_DIST`). For the tests,
configure with `-DLLAMA_BUILD_TESTS=ON` and build `test-backend-ops` and `test-fa-lse`.

## Run

`MODEL=model.gguf scripts/serve-long.sh [ctx] [port]` (defaults 204800 and 8090). What testing showed:

- `--kv-stream` takes `-ctk q8_0` with `-ctv q8_0` (the script's default, for coding) or `-ctv q4_0` (`CTV=q4_0`,
  faster).
- Only models with head size 256 and 6 query heads per KV head stream (the paged kernels' shape). Qwen3.8 27B and
  its derivatives qualify; gpt-oss (head 64), Qwen2.5-Coder (head 128) and Mistral Small 3.x (head 128, 4 query heads
  per KV head) do not yet.
- One model on a 16 GB card at a time.
- `MMPROJ` (vision projector), `TEMPLATE` (chat template file), `SPEC` (drafting arguments) and `POOL_MIB` (decode
  pool) are optional environment variables.

`integrations/systemd/llama-long.service` runs it as a user service. `integrations/pi/ctx-lever.ts` is an extension
for the Pi coding agent: a popup (`/ctx`) that picks model, context window and V cache type, shows what stays on the
GPU and what streams from RAM, writes the service's environment file and restarts it. It reads the attention shape
straight from the GGUF header and refuses models the kernels cannot take, with the reason. Model profiles go in
`~/.config/llama-long/models.json` (example: `long-models.example.json`); the provider names in it are from the
author's setup.

## Open

- The CPU worker for split attention (above), then measuring it at 125K to 250K.
- Intermittent copy stall while reading a long prompt (1 of 4 runs past 160K).
- A bigger prefill pool (`--kv-resident-mib`, now 450) for models with small weights: reading is the slow part at
  depth.
- Port the RDNA4 prompt kernel to Q8_0 V.
- Kernel variants for other attention shapes (head 128 with 4 query heads per KV head; head 256 with 8 to 16).
- Rebase onto current upstream llama.cpp: a trial merge gave 5 conflicting files, 2 trivial and 3 real ports.
- The same measurements on a second machine (an NVIDIA laptop GPU) to test the bandwidth model.
