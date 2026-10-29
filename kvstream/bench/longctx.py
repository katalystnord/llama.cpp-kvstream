#!/usr/bin/env python3
"""Long-context probe for the kv-stream fork (2026-10-28). Builds a prompt of about N tokens from llama.cpp's own source
files (public text), hides a code word at 10% depth, asks for it at the end plus a short task, and prints the server's
prefill and decode speed and whether the code word came back. Usage: longctx.py <port> <target_tokens> [label]"""
import glob, json, os, sys, time, urllib.request

PORT, TARGET = int(sys.argv[1]), int(sys.argv[2])
LABEL = sys.argv[3] if len(sys.argv) > 3 else ''
URL = 'http://127.0.0.1:%d' % PORT
NEEDLE = 'Note for the reader: the code word for the beast test is MARMALADE-7.'
QUESTION = ('\n\nQuestion 1: what is the code word for the beast test, exactly as written above? '
            'Question 2: in about 150 words, what does the file llama-context.cpp above do? Answer both briefly.')


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def ntok(text):
    return len(post('/tokenize', {'content': text})['tokens'])


src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
files = sorted(glob.glob(src + '/src/*.cpp') + glob.glob(src + '/common/*.cpp') + glob.glob(src + '/tools/server/*.cpp'))
first = [f for f in files if f.endswith('src/llama-context.cpp')]
files = first + [f for f in files if f not in first]
parts, total = [], 0
for f in files:
    text = '\n// ===== file %s =====\n' % os.path.relpath(f, src) + open(f, errors='replace').read()
    parts.append(text)
    total += len(text)
    if total > TARGET * 4.2:  # code is about 3 to 4 characters per token; trimmed exactly below
        break
body = ''.join(parts)
toks = post('/tokenize', {'content': body})['tokens']
if len(toks) > TARGET:
    body = post('/detokenize', {'tokens': toks[:TARGET]})['content']
cut = len(body) // 10
body = body[:cut] + '\n\n' + NEEDLE + '\n\n' + body[cut:]
prompt_tokens = ntok(body + QUESTION)

t = time.time()
d = post('/v1/chat/completions', {'model': 'x', 'temperature': 0, 'max_tokens': 400,
                                  'chat_template_kwargs': {'enable_thinking': False},
                                  'messages': [{'role': 'user', 'content': body + QUESTION}]})
wall = time.time() - t
ans = d['choices'][0]['message'].get('content') or ''
tm = d.get('timings', {})
print(json.dumps({'label': LABEL, 'prompt_tokens': prompt_tokens, 'wall_s': round(wall, 1),
                  'prefill_tps': round(tm.get('prompt_per_second', 0), 1), 'decode_tps': round(tm.get('predicted_per_second', 0), 1),
                  'draft': '%s/%s' % (tm.get('draft_n_accepted'), tm.get('draft_n')),
                  'needle_found': 'MARMALADE-7' in ans, 'answer_start': ans[:300]}, ensure_ascii=False))
