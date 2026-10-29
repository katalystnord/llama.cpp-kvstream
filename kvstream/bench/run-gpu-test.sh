#!/bin/bash
# kv-stream fork GPU test (2026-10-28). Qwen3.8 27B IQ3_S, MTP + n-gram
# drafting with ReplaySSM as in the fork's launcher, q8_0 K and q8_0 V (q4 V made Qwen loop in our coding tests).
# Each config: start the fork server, run longctx.py at the given depths, record VRAM and RAM, stop. Log: gpu-test.log
N=$(cd "$(dirname "$0")" && pwd); R=$(cd "$N/../.." && pwd); LS=$R/build-hip/bin/llama-server
M=${MODEL:?set MODEL=path/to/Qwen3.8-27B.gguf (the runs here used the IQ3_S quantisation)}
TPLA=(); [ -n "$TEMPLATE" ] && TPLA=(--chat-template-file "$TEMPLATE")
say() { echo "[$(date +%F\ %T)] $*"; }
vram() { echo $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 )); }
export ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=2 DEBUG_HIP_FORCE_ASYNC_QUEUE=1 DEBUG_HIP_FORCE_GRAPH_QUEUES=1

systemctl --user stop llama-server.service llama-proxy.service 2>/dev/null; sleep 3
[ "$(vram)" -gt 4000 ] && { say "GPU memory in use ($(vram) MiB), not starting"; exit 1; }

run() {  # name ctx extra-args... -- depths...
  local name=$1 ctx=$2; shift 2; local args=(); while [ "$1" != "--" ]; do args+=("$1"); shift; done; shift
  systemd-run --user --scope -q --unit=kvs-$name nice -n 10 $LS -m $M --host 127.0.0.1 --port 8099 -c $ctx -np 1 \
    -ngl 99 -fa on -ctk q8_0 -ctv ${CTV:-q8_0} -ctkd q8_0 -ctvd ${CTV:-q8_0} --jinja "${TPLA[@]}" \
    --spec-type draft-mtp,ngram-mod --spec-rs-replay --spec-draft-n-max 4 --spec-ngram-mod-n-max 32 \
    --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-match 24 -b 2048 -ub 512 "${args[@]}" > $N/server-$name.log 2>&1 </dev/null &
  local s=$(date +%s)
  until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health)" = 200 ] || [ $(( $(date +%s)-s )) -gt 600 ]; do
    systemctl --user is-active -q kvs-$name.scope || { say "$name: server exited, see server-$name.log"; tail -5 $N/server-$name.log; return; }
    sleep 3; done
  say "$name up after $(( $(date +%s)-s )) s: VRAM $(vram) MiB, server RSS $(ps -o rss= -C llama-server | sort -n | tail -1) KiB"
  for d in "$@"; do python3 -I -u $N/longctx.py 8099 $d "$name@$d" || say "$name@$d failed"; say "after $d: VRAM $(vram) MiB"; done
  systemctl --user stop kvs-$name.scope; sleep 5
}

[ -z "$ONLY_STREAM" ] && run base48k  49152  -- 40000
# The fork refuses --kv-stream unless K is q8_0 and V is q4_0 (paged kernels); found on the first run.
export CTV=${CTV_STREAM:-q4_0}
run stream128k 135168 --kv-stream --kv-resident-mib 450 --kv-resident-mib-decode 2420 -- 64000 125000
run stream200k 204800 --kv-stream --kv-resident-mib 450 --kv-resident-mib-decode 2420 -- 195000
say "RUN DONE"
