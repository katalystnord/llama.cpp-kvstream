#!/bin/bash
# Long-context server: a model with the KV cache in host RAM (--kv-stream), MTP + n-gram drafting with ReplaySSM by
# default. Settings as measured on an RX 9070 16 GB (kvstream/README.md): Qwen3.8 27B IQ3_S at 200K context, 195K-token
# prompt read at 497 t/s, answers at 25 t/s, code word recalled.
# --kv-stream takes a q8_0 K cache and a q8_0 (default here, for coding quality) or q4_0 V cache (CTV=q4_0: faster),
# and only models with head size 256 and 6 query heads per KV head (the paged kernels' shape; see pd_shapes_ok).
# Other models through the environment (the Pi context lever writes these to ~/.config/llama-long.env):
#   MODEL=gguf (required)  MMPROJ=vision projector (optional)  TEMPLATE=chat template file (optional; default: the
#   model's own)  SPEC="drafting args"
#   POOL_MIB=GPU memory for cache pages while writing (default 2420; more for models with smaller weights)
# Usage: MODEL=model.gguf kvstream/scripts/serve-long.sh [ctx] [port] [extra llama-server args...]   (defaults 204800
# and 8090). Expects the HIP build in build-hip/ (kvstream/scripts/build-rocm.sh).
R=$(cd "$(dirname "$0")/../.." && pwd)
CTX=${1:-204800}; PORT=${2:-8090}
M=${MODEL:?set MODEL=path/to/model.gguf}
TPL=(); [ -n "$TEMPLATE" ] && TPL=(--chat-template-file "$TEMPLATE")
SPEC=${SPEC:---spec-type draft-mtp,ngram-mod --spec-rs-replay --spec-draft-n-max 4 --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-max 32 --spec-ngram-mod-n-match 24}
VISION=(); [ -n "$MMPROJ" ] && VISION=(--mmproj "$MMPROJ")
export ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=2 DEBUG_HIP_FORCE_ASYNC_QUEUE=1 DEBUG_HIP_FORCE_GRAPH_QUEUES=1
# shellcheck disable=SC2086  # SPEC is a list of arguments
exec "$R/build-hip/bin/llama-server" -m "$M" "${VISION[@]}" --host 127.0.0.1 --port "$PORT" -c "$CTX" -np 1 -ngl 99 -fa on \
  -ctk q8_0 -ctv ${CTV:-q8_0} -ctkd q8_0 -ctvd ${CTV:-q8_0} --kv-stream --kv-resident-mib 450 --kv-resident-mib-decode ${POOL_MIB:-2420} \
  --jinja "${TPL[@]}" $SPEC \
  -b 2048 -ub 512 "${@:3}"  # no --sleep-idle-seconds: the fork's base (2026-10-21) predates it
