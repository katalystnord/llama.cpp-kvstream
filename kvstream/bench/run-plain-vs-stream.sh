#!/bin/bash
# Plain llama.cpp, spilling to RAM, and the stream (2026-10-29), Qwen3.8 27B, MTP drafting only (n-gram drafting keeps
# state across requests and can replay an earlier answer to the same question):
#   plain:   an everyday llama.cpp build (PLAIN_SERVER, here Vulkan), whole cache on the GPU, 48K window;
#   spill 1: the same build with --no-kv-offload (cache in RAM, attention on the CPU), 131K window;
#   spill 2: this fork's HIP build without the stream, GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 (the driver pages GPU memory to
#            RAM), 131K window, capped at 30 minutes per request;
#   stream:  this fork, serve-long.sh.
# Each depth is a fresh prompt (longctx.py). Usage: MODEL=... PLAIN_SERVER=.../llama-server [TEMPLATE=...] run-plain-vs-stream.sh
N=$(cd "$(dirname "$0")" && pwd); R=$(cd "$N/../.." && pwd)
M=${MODEL:?set MODEL}; PS=${PLAIN_SERVER:?set PLAIN_SERVER to an everyday llama-server}
TPL=(); [ -n "$TEMPLATE" ] && TPL=(--chat-template-file "$TEMPLATE")
say() { echo "[$(date +%F\ %T)] $*"; }
vram() { echo $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 )); }
up() { local u=$1 s=$(date +%s)
  until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health)" = 200 ]; do
    systemctl --user is-active -q $u.scope || { say "$u: server exited"; return 1; }; sleep 2; done; }
down() { systemctl --user stop $1.scope; sleep 3; systemctl --user kill -s KILL $1.scope 2>/dev/null; sleep 2; }
systemctl --user stop llama-server.service llama-proxy.service llama-long.service 2>/dev/null
[ "$(vram)" -gt 4000 ] && { say "GPU memory in use ($(vram) MiB), not starting"; exit 1; }
SPEC="--spec-type draft-mtp --spec-draft-n-max 4"

say "plain: $PS, 48K window, q8_0 K and V"
systemd-run --user --scope -q --unit=kvs-plain "$PS" -m "$M" --host 127.0.0.1 --port 8099 -c 49152 -np 1 -ngl 99 -fa on \
  -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 --jinja "${TPL[@]}" $SPEC -b 2048 -ub 512 > "$N/server-plain.log" 2>&1 </dev/null &
if up kvs-plain; then
  say "plain up, VRAM $(vram) MiB"
  for d in 8000 32000 45000; do python3 -I -u "$N/longctx.py" 8099 $d "plain@$d" | sed -E 's/"answer_start".*//'; done
fi
down kvs-plain

say "spill 1: $PS --no-kv-offload, 131K window"
systemd-run --user --scope -q --unit=kvs-nkvo "$PS" -m "$M" --host 127.0.0.1 --port 8099 -c 131072 -np 1 -ngl 99 -fa on \
  --no-kv-offload -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 --jinja "${TPL[@]}" $SPEC -b 2048 -ub 512 > "$N/server-nkvo.log" 2>&1 </dev/null &
if up kvs-nkvo; then
  say "spill 1 up, VRAM $(vram) MiB"
  for d in 64000 125000; do python3 -I -u "$N/longctx.py" 8099 $d "nkvo@$d" | sed -E 's/"answer_start".*//'; done
fi
down kvs-nkvo

say "spill 2: unified memory, HIP build without the stream, 131K window"
systemd-run --user --scope -q --unit=kvs-unified env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ROCR_VISIBLE_DEVICES=0 "$R/build-hip/bin/llama-server" \
  -m "$M" --host 127.0.0.1 --port 8099 -c 131072 -np 1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 \
  --jinja "${TPL[@]}" $SPEC --spec-rs-replay -b 2048 -ub 512 > "$N/server-unified.log" 2>&1 </dev/null &
if up kvs-unified; then
  say "spill 2 up, VRAM $(vram) MiB"
  timeout 1800 python3 -I -u "$N/longctx.py" 8099 125000 "unified@125000" | sed -E 's/"answer_start".*//' || say "unified@125000: no answer within 30 minutes"
fi
down kvs-unified

say "stream: serve-long.sh, 204800 window"
export SPEC="$SPEC --spec-rs-replay" MODEL="$M"
systemd-run --user --scope -q --unit=kvs-stream "$R/kvstream/scripts/serve-long.sh" 204800 8099 > "$N/server-stream.log" 2>&1 </dev/null &
if up kvs-stream; then
  say "stream up, VRAM $(vram) MiB"
  for d in 8000 32000 64000; do python3 -I -u "$N/longctx.py" 8099 $d "stream@$d" | sed -E 's/"answer_start".*//'; done
fi
down kvs-stream
systemctl --user start llama-proxy.socket && say "router socket started again"
say "RUN DONE"
