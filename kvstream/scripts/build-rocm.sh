#!/bin/bash
# Build the kv-stream fork for an RDNA4 GPU (tested: RX 9070, gfx1201) with Ubuntu's ROCm 7.1 packages: HIP via /usr/lib/llvm-21 clang,
# the fork's own cmake options (its .devops/nix/rocm-rx9070xt.nix), plus two fixes for the server's web-UI step:
#   - LLAMA_USE_PREBUILT_UI=OFF: the fork's build number has no prebuilt UI on Hugging Face (download fails);
#   - LLAMA_UI_GZIP=OFF: gzipping the UI files failed across a symlinked source path.
# With the prebuilt UI off, the build embeds whatever is in build-hip/tools/ui/dist; this script copies it from another
# llama.cpp build ($UI_DIST, default: the everyday Vulkan build), since an API server does not need a current UI.
# Usage: kvstream/scripts/build-rocm.sh [targets...]   (default targets: llama-server llama-bench), -j6 at nice 10.
set -e
R=$(cd "$(dirname "$0")/../.." && pwd); B=$R/build-hip
UI_DIST=${UI_DIST:-$HOME/code/scratch/llama.cpp/build/tools/ui/dist}
cmake -S "$R" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_CCACHE=OFF \
  -DCMAKE_HIP_COMPILER=/usr/lib/llvm-21/bin/clang -DGPU_TARGETS=gfx1201 -DAMDGPU_TARGETS=gfx1201 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 "-DGGML_CUDA_FA_QUANTS=q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16,q8_0-q4_0" \
  -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_UI_GZIP=OFF
if [ ! -f "$B/tools/ui/dist/index.html" ]; then
  [ -f "$UI_DIST/index.html" ] || { echo "no UI files at $UI_DIST; set UI_DIST to a built llama.cpp tools/ui/dist" >&2; exit 1; }
  mkdir -p "$B/tools/ui" && cp -r "$UI_DIST" "$B/tools/ui/dist"
fi
[ $# -eq 0 ] && set -- llama-server llama-bench
nice -n 10 cmake --build "$B" -j6 --target "$@"
