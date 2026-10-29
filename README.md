# llama.cpp-kvstream: long context on a 16 GB AMD card

A llama.cpp fork that runs a 27B model at 200K to 250K tokens of context on one 16 GB consumer GPU, by keeping the KV
cache in host RAM and streaming the pages each step needs to the card. This repository is Katalyst Nord's continuation
of that work: more cache formats for the GPU kernels, the first pieces of split attention (the CPU attends over part of
the cache in parallel with the GPU), and the measurements on a cheaper, older platform than the fork was built on.

Measured on an AMD Radeon RX 9070 (non-XT, 16 GB) with a Ryzen 9 5900X, dual-channel DDR4 and PCIe 4.0:

| model | context | read t/s | write t/s | long-range recall |
|---|---|---|---|---|
| Qwen3.8 27B IQ3_S, without streaming | 48K (the most that fits) | 615 | 38.9 | found |
| Qwen3.8 27B IQ3_S, streaming | 195K | 497 | 25.3 | found |
| Saluki 27B (2-bit Qwen3.8), streaming | 250K | 311 | 3.9 | found |

Against putting the cache in RAM and attending on the CPU (`--no-kv-offload`), the stream reads 2.4x and writes 3 to
5.6x faster at 125K. Full tables, logs, scripts and method: [kvstream/README.md](kvstream/README.md).

## Where this comes from

| layer | author | what it brings |
|---|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | the ggml authors | everything else |
| [RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming) | RaymondHuang210129 | adaptive KV streaming, on CUDA |
| [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) | stew675 | RDNA kernel work |
| [achyudh/llama.cpp](https://github.com/achyudh/llama.cpp), branch `feature/adaptive-kv-stream-rocm` | achyudh | the KV stream re-implemented for RDNA4 in ROCm/HIP, with RDNA4 kernels, MTP and n-gram drafting. Its README is kept as [README-kvstream-fork.md](README-kvstream-fork.md) |
| this repository | David Sandquist, Katalyst Nord | the commits below, the `kvstream/` folder |

This repository starts from achyudh's branch at b540a4b9b (2026-10-22). Also thanks to raffidahmad for the Windows
build recipe and to u/bottlestreet3d on Reddit for the report of Qwen3.8 at 262K on an RX 9070 XT.

## What this repository adds

On top of b540a4b9b:

- `kv-stream: Q8_0 V cache for the paged decode kernel`. The fork's decode kernel read a Q4_0 V cache only; Q4_0 V
  made the model loop in coding tasks, Q8_0 V does not.
- `tests: perf cases for decode over a long q8_0 cache`.
- `ggml-cpu: optional per-row log-sum-exp output from flash attention`, with `tests/test-fa-lse.cpp`.
- `ggml-cuda: the paged decode kernel writes the log-sum-exp too`. With the previous commit, a GPU part and a CPU part
  of the attention can be merged exactly.
- `ggml-cpu: grouped-query decode path for flash attention`: each K/V row read once for all query heads that share
  it, 1.5x faster CPU attention at depth.

Branches:

| branch | contents |
|---|---|
| `main` | the commits above and the `kvstream/` folder |
| `cpu-fa-block` | the grouped-query CPU path in blocks of 16 rows; CPU tests pass, not yet measured |

## Status

Work in progress. The next step is the CPU worker inside the stream that completes split attention; the design and
the measurements so far (expected gain about 1.45x at depth on this machine) are in
[kvstream/design/split-attention.md](kvstream/design/split-attention.md). Known limits:

- Only models whose attention has head size 256 and 6 query heads per KV head stream (Qwen3.8 27B and derivatives).
- Linux with ROCm and an RDNA4 GPU (gfx1201) is the only tested path; NVIDIA cards are untested with this fork.
- One intermittent copy stall in 4 deep runs (details in `kvstream/README.md`).
- Based on llama.cpp from 2026-10-21, not rebased onto current upstream.

## Build and run

```bash
kvstream/scripts/build-rocm.sh                           # HIP build into build-hip/
MODEL=/path/to/Qwen3.8-27B.gguf kvstream/scripts/serve-long.sh 204800 8090
python3 kvstream/bench/longctx.py 8090 125000            # the long-context probe
```

Details and options: [kvstream/README.md](kvstream/README.md). For llama.cpp itself, see the
[upstream documentation](https://github.com/ggml-org/llama.cpp) and [docs/build.md](docs/build.md).

## License

MIT, as llama.cpp ([LICENSE](LICENSE)).
