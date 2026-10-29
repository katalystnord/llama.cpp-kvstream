#!/usr/bin/env python3
"""Build data.json (the figure numbers, each with its log) from results/gtx1650-i7-9750h."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(H, 'results', 'gtx1650-i7-9750h')

def rows(name):
    p = os.path.join(R, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

out = []
def add(label, metric, recs, source, log):
    recs = sorted(recs, key=lambda r: r['prompt_tokens'])
    pts = [[r['prompt_tokens'], r[metric]] for r in recs]
    if pts:
        out.append({'machine': 'gtx1650-i7-9750h', 'label': label, 'metric': {'decode_tps': 'write_tps', 'prefill_tps': 'read_tps'}[metric],
                    'points': pts, 'needle_found': [r['needle_found'] for r in recs], 'source': source, 'log': log})

for metric in ('decode_tps', 'prefill_tps'):
    add('plain', metric, rows('plain-43008.jsonl'), 'plain-43008.jsonl', 'server-plain-43008.log')
    add('stream', metric, rows('stream.jsonl'), 'stream.jsonl', 'server-stream.log')
    for n, label, ds in (('spill1', 'spill 1', (32000, 64000)), ('spill2', 'spill 2', (64000,))):  # 125K: cut by the 30-minute cap
        add(label, metric, sum((rows(f'{n}-{d}.jsonl') for d in ds), []), ', '.join(f'{n}-{d}.jsonl' for d in ds),
            ', '.join(f'server-{n}-{d}.log' for d in ds))
    add('stream 250K', metric, rows('stream250.jsonl'), 'stream250.jsonl', 'server-stream250.log')
for mode, label in (('restore', 'split off (slot restored)'), ('split-auto', 'split auto'), ('split-noop', 'split ceiling (output wrong)')):
    recs = rows(f'{mode}-64000.jsonl') + rows(f'{mode}-125000.jsonl')
    add(label, 'decode_tps', recs, f'{mode}-64000.jsonl, {mode}-125000.jsonl', f'server-{mode}-64000.log, server-{mode}-125000.log')
json.dump(out, open(os.path.join(H, 'data.json'), 'w'), indent=1)
print(len(out), 'entries')
