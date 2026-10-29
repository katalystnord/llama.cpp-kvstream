#!/bin/bash
# Split attention settings sweep (2026-10-29): the probe at DEPTH is read once and its slot saved to disk; each setting
# then restarts the server, restores the slot (seconds instead of minutes of reading) and writes one 400-token answer
# at that depth. Settings: "fraction:threads" (0:0 = no split). LLAMA_KV_SPLIT_DEBUG prints where the CPU worker's
# time goes. Default: Saluki, pool 2420 MiB, no drafting, as run-split.sh; MODEL, TEMPLATE and SPEC select another
# model and drafting (the slot file is named after the model and depth).
# Usage: SALUKI_DIR=... | MODEL=... SLOTS=dir [DEPTH=150000] [CONFIGS="0:0 0.2:8 ..."] run-split-sweep.sh   Log: stdout
N=$(cd "$(dirname "$0")" && pwd); R=$(cd "$N/../.." && pwd)
[ -n "$MODEL" ] || MODEL=${SALUKI_DIR:?set SALUKI_DIR to the folder holding Underdog-Saluki-27B-1.0-IQ2-mix.gguf, or MODEL}/Underdog-Saluki-27B-1.0-IQ2-mix.gguf
SLOTS=${SLOTS:?set SLOTS to a folder with room for the slot file (about 5 GB at 150K)}
say() { echo "[$(date +%F\ %T)] $*"; }
vram() { echo $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 )); }
[ "$(vram)" -gt 4000 ] && { say "GPU memory in use ($(vram) MiB), not starting"; exit 1; }
DEPTH=${DEPTH:-150000}; FILE=split-sweep-$(basename "$MODEL" .gguf | cut -c1-24)-$DEPTH.bin
CONFIGS=${CONFIGS:-"0:0 0.2:8 0.4:8 0.2:12 0.4:12 0.3:16"}
export MODEL CTV=q8_0 POOL_MIB=${POOL_MIB:-2420} SPEC=${SPEC:-"--spec-type none"} LLAMA_KV_SPLIT_DEBUG=1
post() { curl -s -m 600 -H 'Content-Type: application/json' -d "$2" "http://127.0.0.1:8099$1"; }

start() {  # name fraction threads
  if [ "$2" = 0 ]; then unset LLAMA_KV_SPLIT; else export LLAMA_KV_SPLIT=$2 LLAMA_KV_SPLIT_THREADS=$3; fi
  systemd-run --user --scope -q --unit=kvs-sweep "$R/kvstream/scripts/serve-long.sh" 163840 8099 --slot-save-path "$SLOTS" \
    > "$N/server-sweep-$1.log" 2>&1 </dev/null &
  until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health)" = 200 ]; do
    systemctl --user is-active -q kvs-sweep.scope || { say "$1: server exited"; tail -5 "$N/server-sweep-$1.log"; return 1; }
    sleep 2; done
}
stop() { systemctl --user stop kvs-sweep.scope; sleep 3; systemctl --user kill -s KILL kvs-sweep.scope 2>/dev/null; sleep 2; }

systemctl --user stop llama-server.service llama-proxy.service llama-long.service 2>/dev/null
if [ ! -s "$SLOTS/$FILE" ]; then
  start fill 0 0 || exit 1
  python3 -I -u "$N/longctx.py" 8099 $DEPTH "fill@$DEPTH"
  say "saved: $(post '/slots/0?action=save' "{\"filename\": \"$FILE\"}" | cut -c1-200)"
  stop
fi
for c in $CONFIGS; do
  f=${c%%:*}; t=${c##*:}; name=f$f-t$t
  start "$name" "$f" "$t" || continue
  say "$name: restored: $(post '/slots/0?action=restore' "{\"filename\": \"$FILE\"}" | cut -c1-160)"
  python3 -I -u "$N/longctx.py" 8099 $DEPTH "$name" | cut -c1-200
  grep 'split:' "$N/server-sweep-$name.log" | tail -2
  stop
done
systemctl --user start llama-proxy.socket && say "router socket started again"
say "SWEEP DONE"
