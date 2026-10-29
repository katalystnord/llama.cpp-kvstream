#!/bin/bash
# Saluki 27B (2-bit Qwen3.8, same attention shape: head 256, GQA 6) on the kv-stream fork, through serve-long.sh with
# the model from the environment as the Pi context lever sets it (2026-10-28). n-gram drafting only (no MTP head),
# q8 K / q8 V, 131K window: the long-context probe at 64K and 125K, then one page image through the projector.
# Usage: [CTX=window] [POOL_MIB=n] [DEPTHS="64000 125000"] run-saluki.sh [image]. Log: stdout; server log: server-saluki.log
N=$(cd "$(dirname "$0")" && pwd); R=$(cd "$N/../.." && pwd)
IMG=${1:?usage: run-saluki.sh <page-image.png>   (any scanned page with a table)}
D=${SALUKI_DIR:?set SALUKI_DIR to the folder holding Underdog-Saluki-27B-1.0-IQ2-mix.gguf and its mmproj}
say() { echo "[$(date +%F\ %T)] $*"; }
vram() { echo $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 )); }
[ "$(vram)" -gt 4000 ] && { say "GPU memory in use ($(vram) MiB), not starting"; exit 1; }

export MODEL=$D/Underdog-Saluki-27B-1.0-IQ2-mix.gguf MMPROJ=$D/mmproj-Underdog-Saluki-27B-1.0-Q8_0.gguf CTV=q8_0
export SPEC="--spec-type ngram-mod --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-max 32 --spec-ngram-mod-n-match 24"
systemd-run --user --scope -q --unit=kvs-saluki nice -n 10 "$R/kvstream/scripts/serve-long.sh" ${CTX:-135168} 8099 > "$N/server-saluki.log" 2>&1 </dev/null &
s=$(date +%s)
until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health)" = 200 ] || [ $(( $(date +%s)-s )) -gt 600 ]; do
  systemctl --user is-active -q kvs-saluki.scope || { say "server exited, see server-saluki.log"; tail -5 "$N/server-saluki.log"; exit 1; }
  sleep 3; done
say "pool ${POOL_MIB:-2420} MiB, up after $(( $(date +%s)-s )) s: VRAM $(vram) MiB, server RSS $(ps -o rss= -C llama-server | sort -n | tail -1) KiB"
for d in ${DEPTHS:-64000 125000}; do python3 -I -u "$N/longctx.py" 8099 $d "saluki@$d" || say "saluki@$d failed"; say "after $d: VRAM $(vram) MiB"; done
python3 -I - "$IMG" <<'EOF'
import base64, json, sys, time, urllib.request
img = base64.b64encode(open(sys.argv[1], 'rb').read()).decode()
body = {'messages': [{'role': 'user', 'content': [
    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + img}},
    {'type': 'text', 'text': 'What kind of page is this? Give its title and, if there is a table, its column headings.'}]}],
    'max_tokens': 1500, 'chat_template_kwargs': {'enable_thinking': False}}
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8099/v1/chat/completions',
    json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=900))
print('image answer (%.1f s, timings %s):' % (time.time() - t, {k: round(v, 1) for k, v in r.get('timings', {}).items() if k.endswith('per_second')}))
print(r['choices'][0]['message']['content'][:1500])
EOF
say "after image: VRAM $(vram) MiB"
systemctl --user stop kvs-saluki.scope
say "RUN DONE"
