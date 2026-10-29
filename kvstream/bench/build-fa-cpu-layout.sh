#!/bin/bash
# Build fa-cpu-layout against the HIP build's ggml libraries (kvstream/scripts/build-rocm.sh builds them).
R=$(cd "$(dirname "$0")/../.." && pwd); B=$R/build-hip/bin
g++ -O2 -std=c++17 -I"$R/ggml/include" "$R/kvstream/bench/fa-cpu-layout.cpp" -o "$B/fa-cpu-layout" \
  -L"$B" -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$B"
