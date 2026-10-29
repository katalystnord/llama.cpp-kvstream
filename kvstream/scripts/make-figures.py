#!/usr/bin/env python3
"""The README figures, from kvstream/docs/figures/data.json: each figure as <name>-light.svg and <name>-dark.svg
(the README picks one by the reader's theme). Usage: kvstream/scripts/make-figures.py"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'docs', 'figures')
DATA = json.load(open(os.path.join(FIG, 'data.json')))

# validated categorical slots (blue, orange, aqua) and text tokens, per theme
THEMES = {
    'light': dict(surface='#fcfcfb', text='#0b0b0b', text2='#52514e', grid='#e4e3df',
                  s1='#2a78d6', s2='#eb6834', s3='#1baf7a'),
    'dark':  dict(surface='#1a1a19', text='#f0efec', text2='#c3c2b7', grid='#383835',
                  s1='#3987e5', s2='#d95926', s3='#199e70'),
}


def axes(t, w=7.2, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(t['surface'])
    ax.set_facecolor(t['surface'])
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(t['grid'])
    ax.tick_params(colors=t['text2'], labelsize=9, length=0)
    ax.grid(True, color=t['grid'], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def title(ax, t, text, sub):
    # at the figure's left edge, above everything (bar labels can reach further left than the plot)
    fig = ax.figure
    fig.text(0.012, 0.975, text, color=t['text'], fontsize=12, fontweight='bold', va='top')
    fig.text(0.012, 0.975 - 0.33/fig.get_figheight(), sub, color=t['text2'], fontsize=9, va='top')


def save(fig, name, theme):
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.62/fig.get_figheight()))
    fig.savefig(os.path.join(FIG, f'{name}-{theme}.svg'), facecolor=fig.get_facecolor())
    plt.close(fig)


def context(t, theme):
    d = DATA['context']
    fig, ax = axes(t, h=4.0)
    title(ax, t, d['title'], 'tokens per second written, MTP drafting, q8_0 cache')
    lim = d['plain']['limit_tokens']/1000
    ax.axvspan(lim, 262, color=t['grid'], alpha=0.45, linewidth=0)
    ax.text(150, 59, 'beyond what plain llama.cpp fits', color=t['text2'], fontsize=8.5, va='top')

    def line(key, color, marker, filled=True):
        xs, ys = zip(*[(x/1000, y) for x, y in d[key]['points']])
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7, zorder=3,
                markerfacecolor=color if filled else t['surface'], markeredgewidth=2,
                linestyle='none' if len(xs) == 1 else '-')
        return xs, ys
    xs, ys = line('stream', t['s1'], 'o')
    for x, y in zip(xs, ys):
        ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points', xytext=(7, 5), color=t['text'], fontsize=8.5)
    ax.text(xs[-1] - 4, ys[-1] + 6, d['stream']['label'], color=t['text2'], fontsize=9, ha='right')
    xs, ys = line('plain', t['s2'], 's')
    for (x, y), off in zip(zip(xs, ys), [(7, 4), (-4, -15), (6, 4)]):
        ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points', xytext=off, color=t['text'], fontsize=8.5)
    ax.text(3, 24, d['plain']['label'], color=t['text2'], fontsize=9)
    xs, ys = line('spill_cpu', t['s3'], 'D')
    ax.annotate(f'{ys[0]:.1f}', (xs[0], ys[0]), textcoords='offset points', xytext=(6, 5), color=t['text'], fontsize=8.5)
    dx, dy = line('spill_driver', t['s3'], 'v', filled=False)
    ax.annotate(f'{ys[-1]:.1f} and {dy[0]:.1f}', (xs[-1], ys[-1]), textcoords='offset points', xytext=(0, 10),
                ha='center', color=t['text'], fontsize=8.5)
    ax.text(162, 1.2, 'spill to RAM: diamonds, CPU attention;\ntriangle, the driver paging GPU memory',
            color=t['text2'], fontsize=8.5, va='bottom')
    ax.set_xlim(0, 262)
    ax.set_ylim(0, 62)
    ax.set_xlabel('context, thousands of tokens', color=t['text2'], fontsize=9)
    save(fig, 'context', theme)


def bars(t, theme, key, name, unit, fmt):
    d = DATA[key]
    fig, ax = axes(t, h=2.4)
    title(ax, t, d['title'], unit)
    labels = [b[0] for b in d['bars']][::-1]
    vals = [b[1] for b in d['bars']][::-1]
    ax.barh(range(len(vals)), vals, height=0.55, color=t['s1'], edgecolor=t['surface'], linewidth=2)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, color=t['text'], fontsize=9)
    for i, v in enumerate(vals):
        ax.text(v + max(vals)*0.01, i, fmt(v), va='center', color=t['text'], fontsize=9)
    ax.set_xlim(0, max(vals)*1.15)
    ax.grid(axis='y', visible=False)
    save(fig, name, theme)


def split(t, theme):
    d = DATA['split']
    fig, ax = axes(t)
    title(ax, t, d['title'], 'tokens per second written against the share of the streamed cache the CPU attends')
    xs, ys = zip(*d['measured']['tps'])
    cx, cy = zip(*d['ceiling']['tps'])
    ax.plot(cx, cy, color=t['s2'], linewidth=2, linestyle=(0, (4, 3)), marker='o', markersize=7, zorder=2)
    ax.text(cx[-1] - 0.01, cy[-1] + 0.5, d['ceiling']['label'], color=t['text2'], fontsize=9, ha='right')
    ax.plot(xs, ys, color=t['s1'], linewidth=2, marker='o', markersize=7, zorder=3)
    ax.text(xs[-1] + 0.02, ys[-1] - 0.3, d['measured']['label'], color=t['text2'], fontsize=9, va='top')
    for x, y in zip(xs, ys):
        ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points', xytext=(0, -16), ha='center', color=t['text'], fontsize=9)
    i = xs.index(d['settled'])
    ax.annotate(f'auto settles at {100*xs[i]:.0f}%: +{100*(ys[i]/ys[0] - 1):.0f}%', (xs[i], ys[i]),
                textcoords='offset points', xytext=(-14, 26), ha='right', color=t['text'], fontsize=9,
                arrowprops=dict(arrowstyle='-', color=t['text2'], linewidth=0.8))
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(0, 18)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xticklabels(['0', '10%', '20%', '30%', '40%', '50%'])
    ax.set_xlabel('CPU share of the streamed pages', color=t['text2'], fontsize=9)
    save(fig, 'split', theme)


for theme, t in THEMES.items():
    context(t, theme)
    bars(t, theme, 'vs_ram', 'vs-ram', 'tokens per second written', lambda v: f'{v:g} t/s')
    split(t, theme)
    bars(t, theme, 'reask', 'reask', 'seconds', lambda v: f'{v:g} s')
print('figures written to', os.path.normpath(FIG))
