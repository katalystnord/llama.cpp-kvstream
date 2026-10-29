#!/bin/bash
# Split attention (2026-10-29): the CPU attends the oldest cold pages in place while the GPU attends the rest, merged by
# log-sum-exp. Saluki long at a depth where most of the cache streams (pool 2420 MiB, as run-contention.sh), no drafting
# (every written token reads the whole cache), with the split on (LLAMA_KV_SPLIT=FRAC) and then off. Per mode: read the
# probe once, then two 400-token answers at that depth at temperature 0 (writing speed). The answers of both modes are
# saved in answers-split/ to compare word for word: the split is exact up to rounding, so they should agree.
# Usage: SALUKI_DIR=... [DEPTH=150000] [FRAC=0.4] [THREADS=8] run-split.sh    Log: stdout; server logs server-split-*.log
N=$(cd "$(dirname "$0")" && pwd); R=$(cd "$N/../.." && pwd)
D=${SALUKI_DIR:?set SALUKI_DIR to the folder holding Underdog-Saluki-27B-1.0-IQ2-mix.gguf}
say() { echo "[$(date +%F\ %T)] $*"; }
vram() { echo $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 )); }
[ "$(vram)" -gt 4000 ] && { say "GPU memory in use ($(vram) MiB), not starting"; exit 1; }
DEPTH=${DEPTH:-150000}; FRAC=${FRAC:-0.4}
export MODEL=$D/Underdog-Saluki-27B-1.0-IQ2-mix.gguf CTV=q8_0 POOL_MIB=2420 SPEC="--spec-type none"
export ANSWER_DIR=$N/answers-split; mkdir -p "$ANSWER_DIR"

run() {  # mode split-fraction
  local mode=$1
  if [ "$2" = 0 ]; then unset LLAMA_KV_SPLIT; else export LLAMA_KV_SPLIT=$2 LLAMA_KV_SPLIT_THREADS=${THREADS:-8}; fi
  systemd-run --user --scope -q --unit=kvs-split-$mode "$R/kvstream/scripts/serve-long.sh" 163840 8099 \
    > "$N/server-split-$mode.log" 2>&1 </dev/null &
  until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health)" = 200 ]; do
    systemctl --user is-active -q kvs-split-$mode.scope || { say "$mode: server exited"; tail -5 "$N/server-split-$mode.log"; return 1; }
    sleep 3; done
  say "$mode: server up, VRAM $(vram) MiB; $(grep -o 'split attention:.*\|LLAMA_KV_SPLIT is set.*' "$N/server-split-$mode.log" | head -1)"
  # longctx.py rebuilds the identical prompt, so after the first call only the question is read again
  python3 -I -u "$N/longctx.py" 8099 $DEPTH "$mode-fill" || say "$mode-fill failed"
  python3 -I -u "$N/longctx.py" 8099 $DEPTH "$mode-write1" || say "$mode-write1 failed"
  python3 -I -u "$N/longctx.py" 8099 $DEPTH "$mode-write2" || say "$mode-write2 failed"
  systemctl --user stop kvs-split-$mode.scope; sleep 3
  systemctl --user kill -s KILL kvs-split-$mode.scope 2>/dev/null; sleep 2
}

systemctl --user stop llama-server.service llama-proxy.service llama-long.service 2>/dev/null
run split "$FRAC"
run nosplit 0
for f in fill write1 write2; do
  cmp -s "$ANSWER_DIR/split-$f.txt" "$ANSWER_DIR/nosplit-$f.txt" && say "$f: answers identical" || say "$f: answers differ"
done
systemctl --user start llama-proxy.socket && say "router socket started again"
say "RUN DONE"
