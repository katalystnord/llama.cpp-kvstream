#!/bin/bash
# The plain way to put the KV cache in RAM, for comparison with --kv-stream (2026-10-28; the question was whether streaming is "almost the same as just
# letting the model spill over into RAM?"). Saluki 27B on the same HIP build, same drafting and probe as run-saluki.sh,
# but --no-kv-offload (cache in host RAM, attention on the CPU) instead of --kv-stream. Log: stdout; server: server-nkvo.log
N=$(cd "$(dirname "$0")" && pwd); R=$(cd "$N/../.." && pwd)
D=${SALUKI_DIR:?set SALUKI_DIR to the folder holding Underdog-Saluki-27B-1.0-IQ2-mix.gguf and its mmproj}
say() { echo "[$(date +%F\ %T)] $*"; }
vram() { echo $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 )); }
# wait for other GPU tests to finish: the script given as WAIT_PID first (its servers come and go), then any test server
while [ -n "$WAIT_PID" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 10; done
while systemctl --user list-units --state=active --no-legend 'moe-*.scope' 'kvs-*.scope' | grep -q .; do sleep 20; done
sleep 5
[ "$(vram)" -gt 4000 ] && { say "GPU memory in use ($(vram) MiB), not starting"; exit 1; }
export ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=2
systemd-run --user --scope -q --unit=kvs-nkvo nice -n 10 "$R/build-hip/bin/llama-server" -m $D/Underdog-Saluki-27B-1.0-IQ2-mix.gguf \
  --host 127.0.0.1 --port 8099 -c 135168 -np 1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --no-kv-offload \
  --jinja ${TEMPLATE:+--chat-template-file "$TEMPLATE"} \
  --spec-type ngram-mod --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-max 32 --spec-ngram-mod-n-match 24 \
  -b 2048 -ub 512 > "$N/server-nkvo.log" 2>&1 </dev/null &
s=$(date +%s)
until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health)" = 200 ] || [ $(( $(date +%s)-s )) -gt 600 ]; do
  systemctl --user is-active -q kvs-nkvo.scope || { say "server exited, see server-nkvo.log"; tail -5 "$N/server-nkvo.log"; exit 1; }
  sleep 3; done
say "no-kv-offload up after $(( $(date +%s)-s )) s: VRAM $(vram) MiB, server RSS $(ps -o rss= -C llama-server | sort -n | tail -1) KiB"
for d in 64000 125000; do timeout 3600 python3 -I -u "$N/longctx.py" 8099 $d "nkvo@$d" || say "nkvo@$d failed or over 1 h"; say "after $d: VRAM $(vram) MiB"; done
systemctl --user kill -s KILL kvs-nkvo.scope
say "RUN DONE"
