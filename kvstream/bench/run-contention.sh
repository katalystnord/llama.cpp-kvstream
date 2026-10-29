#!/bin/bash
# DDR4 contention (2026-10-28, split-attention design): do the GPU's KV stream and CPU attention slow each other down
# when both read host RAM at once? Saluki long at a depth where most of the cache streams (pool 2420 MiB), then:
#   1. writing speed alone (a long answer at depth),  2. CPU attention perf alone (test-backend-ops, 256K q8_0),
#   3. both at once: the same long answer while the CPU benchmark loops.
# Log: stdout. Server log: server-contention.log
N=$(cd "$(dirname "$0")" && pwd); R=$(cd "$N/../.." && pwd)
D=${SALUKI_DIR:?set SALUKI_DIR to the folder holding Underdog-Saluki-27B-1.0-IQ2-mix.gguf and its mmproj}
say() { echo "[$(date +%F\ %T)] $*"; }
vram() { echo $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 )); }
[ "$(vram)" -gt 4000 ] && { say "GPU memory in use ($(vram) MiB), not starting"; exit 1; }
DEPTH=${DEPTH:-150000}
export MODEL=$D/Underdog-Saluki-27B-1.0-IQ2-mix.gguf CTV=q8_0 POOL_MIB=2420
# no drafting: n-gram drafts replay the previous answer to the same question, and every token must stream
export SPEC="--spec-type none"
systemd-run --user --scope -q --unit=kvs-contention "$R/kvstream/scripts/serve-long.sh" 163840 8099 > "$N/server-contention.log" 2>&1 </dev/null &
until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health)" = 200 ]; do
  systemctl --user is-active -q kvs-contention.scope || { say "server exited"; tail -5 "$N/server-contention.log"; exit 1; }; sleep 3; done
say "server up: VRAM $(vram) MiB"
# fill the cache once (the probe); longctx.py rebuilds the identical prompt, so the next two calls reuse the cached
# prefix and measure writing (a 400-token answer at temperature 0) at that depth
python3 -I -u "$N/longctx.py" 8099 $DEPTH "fill@$DEPTH"
BENCH=("$R/build-hip/bin/test-backend-ops" perf -b CPU -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=4,nr23=\[6,1\].*kv=262144,nb=1.*type_K=q8_0,type_V=q8_0')
say "CPU attention alone (us per layer pass, 256K q8_0):"; "${BENCH[@]}" 2>&1 | grep -o "[0-9.]* us/run" | head -1
say "stream alone:"; python3 -I -u "$N/longctx.py" 8099 $DEPTH "alone@$DEPTH"
say "both at once:"
python3 -I -u "$N/longctx.py" 8099 $DEPTH "both@$DEPTH" &
PP=$!
while kill -0 $PP 2>/dev/null; do "${BENCH[@]}" 2>&1 | grep -o "[0-9.]* us/run" | head -1; done > "$N/contention-bench.txt"
wait $PP
say "CPU attention during the stream (us per layer pass, one line per benchmark run):"; cat "$N/contention-bench.txt"
systemctl --user kill -s KILL kvs-contention.scope
say "RUN DONE"
