#!/usr/bin/env python3
"""Read once, ask later (2026-10-28): does the kv-stream server save a read document to disk and load it back fast?
Steps: read a long document (prompt ends exactly at the document's end, so the hybrid model's recurrent state is at
that point), save slot 0 to a file, ask question 1 (only the question should be read), then the caller restarts the
server and runs this again with --restore: load the file, ask question 2. Prompts are token lists, so the document's
tokens are identical in every request (text joined at the boundary could tokenise differently).
Usage: save_restore.py <port> <target_tokens> <filename> [--restore]"""
import glob, json, os, sys, time, urllib.request

PORT, TARGET, FNAME = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
RESTORE = '--restore' in sys.argv
URL = 'http://127.0.0.1:%d' % PORT
NEEDLE = 'Note for the reader: the code word for the beast test is MARMALADE-7.'
SUFFIX_END = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'  # thinking off, as the chat template does
Q1 = '\n\nIn about 80 words, what does the file llama-context.cpp above do?'
Q2 = '\n\nWhat is the code word for the beast test, exactly as written above? Then name three files from the text above.'


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def tok(text):
    return post('/tokenize', {'content': text, 'add_special': False, 'parse_special': True})['tokens']


def document():
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    files = sorted(glob.glob(src + '/src/*.cpp') + glob.glob(src + '/common/*.cpp') + glob.glob(src + '/tools/server/*.cpp'))
    first = [f for f in files if f.endswith('src/llama-context.cpp')]
    parts, total = [], 0
    for f in first + [f for f in files if f not in first]:
        t = '\n// ===== file %s =====\n' % os.path.relpath(f, src) + open(f, errors='replace').read()
        parts.append(t); total += len(t)
        if total > TARGET * 4.2:
            break
    body = ''.join(parts)
    toks = tok(body)
    if len(toks) > TARGET:
        body = post('/detokenize', {'tokens': toks[:TARGET]})['content']
    cut = len(body) // 10
    return '<|im_start|>user\n' + body[:cut] + '\n\n' + NEEDLE + '\n\n' + body[cut:]


def ask(prefix, question, label):
    t = time.time()
    d = post('/completion', {'prompt': prefix + tok(question + SUFFIX_END), 'n_predict': 250, 'temperature': 0,
                             'cache_prompt': True, 'id_slot': 0})
    tm = d.get('timings', {})
    print(json.dumps({'step': label, 'wall_s': round(time.time() - t, 1), 'tokens_read': tm.get('prompt_n'),
                      'read_ms': round(tm.get('prompt_ms', 0)), 'write_tps': round(tm.get('predicted_per_second', 0), 1),
                      'needle_found': 'MARMALADE-7' in d.get('content', ''), 'answer': d.get('content', '')[:240]},
                     ensure_ascii=False), flush=True)


prefix = tok(document())
print(json.dumps({'document_tokens': len(prefix)}), flush=True)
if not RESTORE:
    t = time.time()
    d = post('/completion', {'prompt': prefix, 'n_predict': 0, 'cache_prompt': True, 'id_slot': 0})
    print(json.dumps({'step': 'read document', 'wall_s': round(time.time() - t, 1),
                      'read_tps': round(d.get('timings', {}).get('prompt_per_second', 0), 1)}), flush=True)
    t = time.time()
    s = post('/slots/0?action=save', {'filename': FNAME})
    print(json.dumps({'step': 'save', 'wall_s': round(time.time() - t, 1), 'server': s}), flush=True)
    ask(prefix, Q1, 'question 1 (same server)')
else:
    t = time.time()
    r = post('/slots/0?action=restore', {'filename': FNAME})
    print(json.dumps({'step': 'restore', 'wall_s': round(time.time() - t, 1), 'server': r}), flush=True)
    ask(prefix, Q2, 'question 2 (fresh server, restored)')
