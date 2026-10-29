#!/bin/bash
# Read once, ask later: read a long document on the kv-stream server, save the slot to disk, ask; restart the server
# cold, restore the slot, ask again. Log: run-save-restore.log next to this script. Slot files: $SLOTS (default ~/llm-cache/slots).
N=$(cd "$(dirname "$0")" && pwd); R=$(cd "$N/../.." && pwd)
DEPTH=${DEPTH:-125000}; SLOTS=${SLOTS:-$HOME/llm-cache/slots}; mkdir -p "$SLOTS"
say() { echo "[$(date +%F\ %T)] $*"; }
vram() { echo $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 )); }
systemctl --user stop llama-server.service llama-proxy.service 2>/dev/null; sleep 3
[ "$(vram)" -gt 4000 ] && { say "GPU memory in use ($(vram) MiB), not starting"; exit 1; }
start() {
  systemd-run --user --scope -q --unit=kvs-saverestore "$R/kvstream/scripts/serve-long.sh" 204800 8099 --slot-save-path "$SLOTS" \
    > "$N/server-saverestore-$1.log" 2>&1 </dev/null &
  local s=$(date +%s)
  until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health)" = 200 ] || [ $(( $(date +%s)-s )) -gt 600 ]; do
    systemctl --user is-active -q kvs-saverestore.scope || { say "server exited"; tail -5 "$N/server-saverestore-$1.log"; exit 1; }
    sleep 3; done
  say "server up ($1) after $(( $(date +%s)-s )) s, VRAM $(vram) MiB"
}
start first
python3 -I -u "$N/save_restore.py" 8099 "$DEPTH" "doc-$DEPTH.bin"
ls -l "$SLOTS"
systemctl --user stop kvs-saverestore.scope; sleep 5
start second
python3 -I -u "$N/save_restore.py" 8099 "$DEPTH" "doc-$DEPTH.bin" --restore
systemctl --user stop kvs-saverestore.scope
say "RUN DONE"
