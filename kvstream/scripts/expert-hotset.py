#!/usr/bin/env python3
"""Routing skew and hot-set report for a MoE model, from a stock llama-imatrix GGUF.

llama-imatrix stores, for each expert tensor, `<name>.counts` with one count per expert:
the number of tokens routed to that expert over the calibration text. This script reads
those counts and answers: how skewed is routing per layer, and what fraction of routed
expert reads would a VRAM hot set of a given size serve (static placement, best case)?

Usage:
  expert-hotset.py IMATRIX.gguf [--model MODEL.gguf] [--budget-mib N ...] [--hotset-out FILE.json]

With --model, expert sizes come from the model's tensors (all *_exps tensors of a layer).
With --budget-mib, the slots are shared across layers greedily by count, which is the best
static split: per-layer sorted counts are non-increasing, so taking the largest remaining
count each time maximises the total hits for a fixed number of slots.
"""

import argparse
import heapq
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402

LAYER_RE = re.compile(r"^blk\.(\d+)\.(ffn_\w+_exps)\.weight\.counts$")


def read_counts(path):
    """Return {layer: counts[n_expert]} taken from one expert tensor kind per layer."""
    r = gguf.GGUFReader(path)
    by_kind = {}
    for t in r.tensors:
        m = LAYER_RE.match(t.name)
        if m:
            by_kind.setdefault(m.group(2), {})[int(m.group(1))] = np.asarray(t.data, dtype=np.float64).reshape(-1)
    if not by_kind:
        sys.exit(f"{path}: no blk.N.ffn_*_exps.weight.counts tensors (not an imatrix GGUF of a MoE model?)")
    # every expert tensor of a layer sees the same routing; prefer down, then whatever exists
    for kind in ("ffn_down_exps", "ffn_up_exps", "ffn_gate_up_exps", "ffn_gate_exps"):
        if kind in by_kind:
            return kind, by_kind[kind]
    kind = sorted(by_kind)[0]
    return kind, by_kind[kind]


def expert_bytes_per_layer(model_path):
    """Return {layer: bytes of one expert summed over all *_exps tensors of that layer}."""
    r = gguf.GGUFReader(model_path)
    out = {}
    for t in r.tensors:
        m = re.match(r"^blk\.(\d+)\.ffn_\w+_exps\.weight$", t.name)
        if m:
            n_exp = int(t.shape[-1])
            out[int(m.group(1))] = out.get(int(m.group(1)), 0) + int(t.n_bytes) // n_exp
    return out


def gini(x):
    x = np.sort(x)
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def k_for_share(sorted_desc, share):
    c = np.cumsum(sorted_desc) / sorted_desc.sum()
    return int(np.searchsorted(c, share) + 1)


def greedy_split(counts, slot_cost, budget_bytes):
    """Fill the budget with the experts that have the highest count per byte, across all layers."""
    order = {l: np.argsort(-c, kind="stable") for l, c in counts.items()}
    taken = {l: 0 for l in counts}
    heap = [(-counts[l][order[l][0]] / slot_cost[l], l) for l in counts]
    heapq.heapify(heap)
    used = 0
    while heap:
        _, l = heapq.heappop(heap)
        if used + slot_cost[l] > budget_bytes:
            continue
        used += slot_cost[l]
        taken[l] += 1
        if taken[l] < len(order[l]):
            e = order[l][taken[l]]
            heapq.heappush(heap, (-counts[l][e] / slot_cost[l], l))
    hot = {l: [int(e) for e in order[l][: taken[l]]] for l in counts}
    hits = sum(counts[l][hot[l]].sum() for l in counts)
    total = sum(c.sum() for c in counts.values())
    return hot, used, hits / total if total else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imatrix")
    ap.add_argument("--model", help="model GGUF, for expert sizes")
    ap.add_argument("--budget-mib", type=float, nargs="*", default=[], help="VRAM budgets for experts, MiB")
    ap.add_argument("--hotset-out", help="write {layer: [expert ids, hottest first]} for the last budget")
    a = ap.parse_args()

    kind, counts = read_counts(a.imatrix)
    layers = sorted(counts)
    n_exp = len(counts[layers[0]])
    total = sum(c.sum() for c in counts.values())
    print(f"{a.imatrix}: {len(layers)} MoE layers, {n_exp} experts, counts from {kind}, {int(total)} routed reads")
    print()
    print("layer  gini  top10%  k@50%  k@80%  k@90%  k@95%  unused")
    for l in layers:
        c = counts[l]
        s = np.sort(c)[::-1]
        top10 = s[: max(1, n_exp // 10)].sum() / s.sum() if s.sum() else 0
        print(f"{l:5d}  {gini(c):.2f}  {top10:6.1%}  {k_for_share(s, .5):5d}  {k_for_share(s, .8):5d}"
              f"  {k_for_share(s, .9):5d}  {k_for_share(s, .95):5d}  {int((c == 0).sum()):6d}")

    if not a.budget_mib:
        return
    if not a.model:
        sys.exit("--budget-mib needs --model for expert sizes")
    cost = expert_bytes_per_layer(a.model)
    missing = [l for l in layers if l not in cost]
    if missing:
        sys.exit(f"model has no *_exps tensors for layers {missing}")
    all_bytes = sum(cost[l] * n_exp for l in layers)
    print()
    print(f"all experts: {all_bytes / 2**20:.0f} MiB; one expert (all its tensors) in layer {layers[0]}: {cost[layers[0]] / 2**20:.2f} MiB")
    print("budget MiB  experts held  share held  static hit rate")
    hot = None
    for b in a.budget_mib:
        hot, used, hit = greedy_split(counts, cost, b * 2**20)
        n_held = sum(len(v) for v in hot.values())
        print(f"{b:10.0f}  {n_held:12d}  {used / all_bytes:10.1%}  {hit:15.1%}")
    if a.hotset_out and hot is not None:
        Path(a.hotset_out).write_text(json.dumps({str(l): v for l, v in hot.items()}, indent=1) + "\n")
        print(f"\nwrote {a.hotset_out}")


if __name__ == "__main__":
    main()
