# brick: GTX 1650 Max-Q 4 GB, i7-9750H

## The machine

| | |
|---|---|
| Laptop | Lenovo ThinkPad X1 Extreme Gen 2 (20QV001FMX, BIOS 1.57) |
| GPU | NVIDIA GeForce GTX 1650 with Max-Q Design, 4 GB, TU117, sm_75, VBIOS 90.17.31.00.26 |
| GPU power limit | 35 W (default, min 1, max 35: fixed by the firmware) |
| GPU link | PCIe 3.0 x16 |
| Display | runs on the Intel UHD 630, so all 4 GB of the NVIDIA card are free for the model |
| CPU | Intel Core i7-9750H, 6 cores, 12 threads, up to 4.5 GHz, AVX2 and FMA |
| RAM | 32 GB: two 16 GB DDR4 SO-DIMMs (RMSA3300ME78HBF-2666), dual channel, 2667 MT/s (42.7 GB/s theoretical) |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic |
| Driver, CUDA | NVIDIA 595.99.02, CUDA toolkit 12.4 (V12.4.131), gcc 15.2.0, cmake 4.2.3 |

## The code

RaymondHuang's adaptive KV streaming fork of llama.cpp at bcf1a81d6, plus the six patches in `patches/` (branch
brick/split-attention):

1. build option GGML_CUDA_KV_STREAM_D512 (head-512 units left out: they need more than 24 GB RAM each to compile);
2. CPU flash attention with LSE output and the grouped-query decode path (from beast);
3. cached KV host store when LLAMA_KV_SPLIT is set;
4. split attention (LLAMA_KV_SPLIT=fraction, LLAMA_KV_SPLIT_VERIFY);
5. LLAMA_KV_SPLIT=auto and LLAMA_KV_SPLIT_NOOP;
6. test-fa-lse and the measurement scripts (scripts/kv-stream-brick).

## The runs

Model: Qwen3.5 4B Q4_K_M (8 full-attention layers of 32, 16 query heads, 4 KV heads of 256, no MTP head, so no
drafting). KV cache q8_0 K and q8_0 V for the release set (earlier split sweeps: q8_0 K, q4_0 V; each table says
which). Greedy, batch and ubatch 256, flash attention on, one slot.

## How to reproduce

```
git clone https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming llama.cpp-adaptive-kv-streaming
cd llama.cpp-adaptive-kv-streaming
git checkout bcf1a81d6
git am ../patches/*.patch
cmake -B build-sm75 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA_KV_STREAM_D512=OFF \
  -DGGML_CUDA_FA_QUANTS="f16-f16;q4_0-q4_0;q8_0-q4_0;q8_0-q8_0;q5_1-q5_1"
cmake --build build-sm75 --target llama-server test-fa-lse --parallel 3
```

Streaming server, as measured (131K window, 896 MiB GPU arena):

```
build-sm75/bin/llama-server -m Qwen3.5-4B-Q4_K_M.gguf --ctx-size 131072 --parallel 1 --n-gpu-layers 999 \
  --flash-attn on --fit off --cache-type-k q8_0 --cache-type-v q8_0 --no-mmproj --spec-type none -b 256 -ub 256 \
  --shared-device-memory-mib 896
```

Add `LLAMA_KV_SPLIT=auto LLAMA_KV_SPLIT_THREADS=6` to the environment for split attention. The whole measurement
set: scripts/kv-stream-brick/measure.sh (see its README for the folder layout it expects).
