#!/bin/bash
# Copy stall reproduction (2026-10-29). The stall seen on 2026-10-28 and twice on 2026-10-29: while reading a long
# prompt at a 256K window, one layer's page copy (layer 9 of the ubatch at about 166K tokens, tag 298) never reports
# done and the 30 s watchdog aborts; other runs with the same settings pass. Here the watchdog waits 15 minutes
# (LLAMA_KV_STREAM_TIMEOUT=900, or TIMEOUT=seconds) and the server logs at verbosity 4 (shows the SDMA engine), so a stall shows whether the
# copy is slow or lost. Up to ATTEMPTS reads of the 250K probe (Saluki, pool 2420 MiB, no drafting), stopping at the
# first stall. Usage: SALUKI_DIR=... [ATTEMPTS=3] repro-stall.sh   Log: stdout; server logs server-repro-<n>.log
N=$(cd "$(dirname "$0")" && pwd); R=$(cd "$N/../.." && pwd)
D=${SALUKI_DIR:?set SALUKI_DIR}
say() { echo "[$(date +%F\ %T)] $*"; }
vram() { echo $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 )); }
export MODEL=$D/Underdog-Saluki-27B-1.0-IQ2-mix.gguf CTV=q8_0 POOL_MIB=2420 SPEC="--spec-type none" LLAMA_KV_STREAM_TIMEOUT=${TIMEOUT:-900}
systemctl --user stop llama-server.service llama-proxy.service llama-long.service 2>/dev/null
for a in $(seq 1 ${ATTEMPTS:-3}); do
  [ "$(vram)" -gt 4000 ] && { say "GPU memory in use ($(vram) MiB), not starting"; exit 1; }
  systemd-run --user --scope -q --unit=kvs-repro "$R/kvstream/scripts/serve-long.sh" 262144 8099 -lv 4 > "$N/server-repro-$a.log" 2>&1 </dev/null &
  until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health)" = 200 ]; do
    systemctl --user is-active -q kvs-repro.scope || { say "attempt $a: server exited"; break; }; sleep 3; done
  say "attempt $a: $(grep -m1 -o 'host to device copies on SDMA engine.*' "$N/server-repro-$a.log")"
  python3 -I -u "$N/longctx.py" 8099 250000 "repro-$a" | sed -E 's/"answer_start".*//' || say "attempt $a: probe failed"
  stalled=$(grep -c 'stalled for' "$N/server-repro-$a.log")
  say "attempt $a: stall messages $stalled; $(grep -m3 'stalled for\|log_state: tag' "$N/server-repro-$a.log" | tr '\n' ' ' | cut -c1-300)"
  systemctl --user stop kvs-repro.scope; sleep 3; systemctl --user kill -s KILL kvs-repro.scope 2>/dev/null; sleep 3
  journalctl -k --since "-20 min" --no-pager 2>/dev/null | grep -iE 'sdma|page fault|ring .*timeout|gpu reset' | head -5
  [ "$stalled" -gt 0 ] && { say "stalled on attempt $a, stopping"; break; }
done
systemctl --user start llama-proxy.socket && say "router socket started again"
say "REPRO DONE"
