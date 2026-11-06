"""Builds the plain-language findings page (Secret Sauce report recipe) from the sim results."""
import json, glob, html
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/tmp/claude-1000/-home-davidhatley-Projects/75e3e499-13e2-4f0d-8132-fa574d6af4d8/scratchpad/report_base.html")
OUT = Path("/tmp/claude-1000/-home-davidhatley-Projects/75e3e499-13e2-4f0d-8132-fa574d6af4d8/scratchpad/kv-eviction-findings.html")

S = [json.load(open(f)) for f in glob.glob(str(ROOT / "data/sim/*.json")) if Path(f).stem not in ("exact_check", "failures")]
EX = json.load(open(ROOT / "data/sim/exact_check.json"))
EXR = [x["greedy"] / x["lp"] for x in EX]
FAIL = json.load(open(ROOT / "data/sim/failures.json"))["0.1"]
TASK_OF = {json.loads(l)["id"]: json.loads(l)["task"] for l in open(ROOT / "data/tasks.jsonl")}
LIVE = {Path(f).stem: json.load(open(f)) for f in glob.glob(str(ROOT / "data/live/*.json"))}
def live(task, b, k):
    v = [r[f"{float(b)}|{k}"]["score"] for pid, r in LIVE.items() if TASK_OF[pid] == task and f"{float(b)}|{k}" in r]
    return (float(np.mean(v)), len(v)) if v else (None, 0)
def live_full(task):
    v = [r["full"]["score"] for pid, r in LIVE.items() if TASK_OF[pid] == task and "full" in r]
    return (float(np.mean(v)), len(v)) if v else (None, 0)
def agg(task, b, k):
    ps = [s for s in S if s["meta"]["task"] == task]
    return float(np.mean([sum(s[b][k]) / sum(s[b]["oracle_w"]) for s in ps])), len(ps)
def ofull(task, b):
    ps = [s for s in S if s["meta"]["task"] == task]
    return float(np.mean([sum(s[b]["oracle_w"]) / sum(s[b]["full"]) for s in ps]))
NS = {t: agg(t, "0.1", "h2o")[1] for t in ["needle", "summ", "mdqa"]}

TASKS = [("needle", "Needle in a haystack"), ("summ", "Summarization"), ("mdqa", "Multi-doc QA")]
POL = [("random", "Random"), ("lru", "LRU"), ("streaming", "StreamingLLM"), ("h2o", "H2O"), ("snapkv", "SnapKV"), ("belady@0.01", "Belady oracle")]

# ---------- plate 1: bars at 10% budget, one panel per task
def plate1():
    W, rowh, top, left, pw = 300, 22, 30, 104, 170
    panels = []
    for task, name in TASKS:
        H = top + rowh * len(POL) + 30
        x = lambda v: left + v * pw
        g = [f'<text class="lb" x="0" y="14">{html.escape(name.upper())}</text>']
        for i, (k, lbl) in enumerate(POL):
            v = agg(task, "0.1", k)[0]
            y = top + i * rowh
            cls = "b b--ghost" if k.startswith("belady") else "b"
            g.append(f'<text class="tk" x="0" y="{y+13}">{lbl}</text>')
            g.append(f'<rect class="{cls}" x="{left}" y="{y+2}" width="{v*pw:.1f}" height="14"/>')
            g.append(f'<text class="vl" x="{x(v)+6:.1f}" y="{y+13}">{v:.2f}</text>')
        yb = top + rowh * len(POL) + 4
        g.append(f'<path class="ax" d="M{left}.5 {top-4}V{yb}"/><path class="ref" d="M{x(1):.1f} {top-4}V{yb}"/>')
        g.append(f'<text class="tk" x="{x(1)-4:.1f}" y="{yb+16}" text-anchor="end">1.0 = oracle</text>')
        g.append(f'<text class="tk" x="{left}" y="{yb+16}">0</text>')
        panels.append(f'<svg class="chart" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
                      f'aria-label="{name}: share of the oracle\'s attention kept at a 10% budget">{"".join(g)}</svg>')
    return "".join(panels)

# ---------- plate 2: best heuristic vs budget, three tasks as lines
def plate2():
    W, H, l, r, t, b = 560, 230, 48, 150, 16, 40
    bud = ["0.05", "0.1", "0.25", "0.5"]
    xs = [l + i * (W - l - r) / 3 for i in range(4)]
    lo, hi = 0.75, 1.0
    y = lambda v: t + (hi - v) / (hi - lo) * (H - t - b)
    g = []
    for v in [0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
        g.append(f'<path class="grid" d="M{l} {y(v):.1f}H{W-r}"/><text class="tk" x="{l-8}" y="{y(v)+4:.1f}" text-anchor="end">{v:.2f}</text>')
    for xi, bb in zip(xs, bud):
        g.append(f'<text class="tk" x="{xi:.1f}" y="{H-b+18}" text-anchor="middle">{float(bb):.0%}</text>')
    g.append(f'<text class="tk" x="{(l+W-r)/2:.1f}" y="{H-4}" text-anchor="middle">cache budget (share of the prompt kept)</text>')
    styles = {"needle": "ln", "summ": "ln ln--dash", "mdqa": "ln ln--ink"}
    for task, name in TASKS:
        vals = [max(agg(task, bb, k)[0] for k in ["streaming", "h2o", "snapkv"]) for bb in bud]
        pts = " ".join(f"{xi:.1f},{y(v):.1f}" for xi, v in zip(xs, vals))
        g.append(f'<polyline class="{styles[task]}" points="{pts}"/>')
        for xi, v in zip(xs, vals):
            g.append(f'<circle class="dot {styles[task].split()[-1]}" cx="{xi:.1f}" cy="{y(v):.1f}" r="4"/>')
        g.append(f'<text class="tk" x="{xs[-1]+10:.1f}" y="{y(vals[-1])+4 + (12 if task=="mdqa" else (-10 if task=="needle" else 0)):.1f}">{name}</text>')
    return f'<svg class="chart chart--wide" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Best heuristic, share of oracle attention kept, at budgets 5 to 50 percent">{"".join(g)}</svg>'

# ---------- plate 3: where needle attention goes (stacked bar)
def plate3():
    parts = [("4 sink tokens", 0.33, "seg seg--a"), ("last 64 prompt tokens", 0.29, "seg seg--b"), ("all other ~11,000 tokens", 0.38, "seg seg--c")]
    W, x0, w = 560, 0, 560
    g, x = [], x0
    for i, (lbl, v, c) in enumerate(parts):
        g.append(f'<rect class="{c}" x="{x:.1f}" y="8" width="{v*w-2:.1f}" height="28"/>')
        g.append(f'<text class="tk" x="{x:.1f}" y="{56 + 16 * i}">{v:.0%} · {lbl}</text>')
        x += v * w
    return f'<svg class="chart chart--wide" viewBox="0 0 {W} 98" width="{W}" height="98" role="img" aria-label="Needle task: 33% of attention on 4 sink tokens, 29% on the last 64 prompt tokens, 38% on everything else">{"".join(g)}</svg>'

# ---------- plate 4: needle-token attention kept (single panel bars, 10% budget)
def plate4():
    W, rowh, top, left, pw = 560, 22, 8, 120, 360
    items = [("Weighted oracle", FAIL["oracle_w"]["needle_mass"])] + [(lbl, FAIL[k]["needle_mass"]) for k, lbl in
             [("snapkv", "SnapKV"), ("lru", "LRU"), ("h2o", "H2O"), ("streaming", "StreamingLLM")]]
    fn = FAIL["full_needle_mass"]
    g = []
    for i, (lbl, m) in enumerate(items):
        v, y = m / fn, top + i * rowh
        cls = "b b--ghost" if i == 0 else "b"
        g.append(f'<text class="tk" x="0" y="{y+13}">{lbl}</text>')
        g.append(f'<rect class="{cls}" x="{left}" y="{y+2}" width="{max(v*pw, 1):.1f}" height="14"/>')
        g.append(f'<text class="vl" x="{left + v*pw + 6:.1f}" y="{y+13}">{v:.2f}</text>')
    yb = top + rowh * len(items) + 4
    g.append(f'<path class="ax" d="M{left}.5 {top-4}V{yb}"/><path class="ref" d="M{left+pw} {top-4}V{yb}"/>')
    g.append(f'<text class="tk" x="{left+pw-4}" y="{yb+16}" text-anchor="end">1.0 = full cache</text>')
    H = yb + 22
    return f'<svg class="chart chart--wide" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Share of the attention on the hidden number that each policy keeps at a 10% budget">{"".join(g)}</svg>'

LIVEPOL = [("snapkv", "SnapKV"), ("h2o", "H2O"), ("lru", "LRU"), ("streaming", "StreamingLLM"), ("random", "Random"), ("oracle_replay", "Oracle, run live")]
def livetable(task):
    f, n = live_full(task)
    if not n:
        return ""
    rows = ""
    for k, lbl in LIVEPOL:
        rows += f"<tr><td>{lbl}</td>" + "".join(f'<td class="num">{live(task, b, k)[0]:.2f}</td>' for b in ["0.05", "0.1", "0.25", "0.5"]) + "</tr>"
    return rows + f'<tr><td><em>Full cache, no eviction</em></td><td class="num" colspan="4">{f:.2f}</td></tr>'

def row(task, k):
    return "".join(f'<td class="num">{agg(task, b, k)[0]:.3f}</td>' for b in ["0.05", "0.1", "0.25", "0.5"])

numtable = ""
for task, name in TASKS:
    for k, lbl in POL[1:]:
        numtable += f"<tr><td>{name}{' (6 of 10 prompts)' if NS[task] < 10 else ''}</td><td>{lbl}</td>{row(task, k)}</tr>"
belsweep = ""
for task, name in TASKS:
    belsweep += f"<tr><td>{name}</td>" + "".join(f'<td class="num">{agg(task, "0.1", f"belady@{x}")[0]:.2f}</td>' for x in [0.001, 0.003, 0.01, 0.03]) + "</tr>"
oracle_full = " · ".join(f"{name}: {ofull(t, '0.05'):.0%}" for t, name in TASKS)

base = BASE.read_text()
head = base[base.index("<!-- sauce-lint surface: docs -->"): base.index("<style>\n/* layout only */")]
page = f'''<title>KV Cache Oracle Gap</title>
<meta name="claude:viewport-fit-sides" content="cover">
{head}<style>
/* layout only: report recipe; summary grid, plates of SVG charts drawn 1:1 */
.summary {{ display: grid; grid-template-columns: minmax(0, 1fr) 17rem; gap: var(--space-6); align-items: start; }}
.summary > * {{ min-width: 0; }}
.figures {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--space-4); margin: 0; }}
.figures div {{ border-top: var(--line-heavy) solid var(--ink); padding-top: var(--space-2); }}
.figures dt {{ font: var(--type-label); letter-spacing: .14em; text-transform: uppercase; color: var(--ink-muted); }}
.figures dd {{ margin: 0; color: var(--cobalt-text); }}
.panels {{ display: flex; flex-wrap: wrap; gap: var(--space-6); }}
.chart {{ display: block; max-width: 100%; height: auto; }}
.chart .b {{ fill: var(--cobalt); }} .chart .b--ghost {{ fill: none; stroke: var(--ink); stroke-width: 1; stroke-dasharray: 3 2; }}
.chart .ax, .chart .ref {{ stroke: var(--ink); stroke-width: 1; }} .chart .ref {{ stroke-dasharray: 2 3; }}
.chart .grid {{ stroke: var(--rule); stroke-width: 1; }}
.chart .lb {{ fill: var(--ink); font-family: var(--font-mono); font-size: 11px; font-weight: 600; letter-spacing: .14em; }}
.chart .tk, .chart .vl {{ fill: var(--ink); font-family: var(--font-mono); font-size: 11px; font-weight: 400; }}
.chart .ln {{ fill: none; stroke: var(--cobalt); stroke-width: 2; }} .chart .ln--dash {{ stroke-dasharray: 6 4; }} .chart .ln--ink {{ stroke: var(--ink); }}
.chart .dot {{ fill: var(--cobalt); }} .chart .dot.ln--ink {{ fill: var(--ink); }}
.chart .seg--a {{ fill: var(--ink); }} .chart .seg--b {{ fill: var(--cobalt); }} .chart .seg--c {{ fill: none; stroke: var(--ink); stroke-width: 1; }}
.stack {{ display: grid; gap: var(--space-6); }}
.glossary {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-4) var(--space-6); margin: 0; }}
.glossary > div {{ min-width: 0; border-top: var(--line-hair) solid var(--rule); padding-top: var(--space-2); }}
.glossary dd {{ margin: 0; }}
@media (max-width: 760px) {{ .summary, .glossary {{ grid-template-columns: minmax(0, 1fr); }} }}
</style>
<div class="ss-body">
<div class="ss-page stack">
  <header class="ss-runhead"><span><b>Research log</b> · kvoracle · Qwen2.5-7B-Instruct</span><span>2026-11-06 · in progress</span></header>

  <h1 class="ss-mast ss-mast--m"><span class="ss-mast__x">KV Cache</span><span class="ss-mast__c">Oracle Gap</span></h1>

  <div class="summary">
    <div class="ss-prose">
      <p><strong>Short answer so far: by the attention they keep, the best simple eviction rules get about 89–95% of what a perfect-hindsight oracle keeps, at a cache budget of 10% of the prompt. But that number hides the part that matters. When the rules run for real, only one of them (SnapKV) still finds a hidden number in a long document; the other three get 0 out of 10.</strong></p>
      <p>The question is how much a language model loses when it is forced to forget most of what it has read, and how close today's forgetting rules come to the best possible forgetting. We measured this on 30 long prompts (8,000 to 16,000 tokens each) across three kinds of task.</p>
      <p>Three results stand out. First, attention kept does not predict answers kept: rules that keep 84–88% of the oracle's attention fail retrieval completely, because what they drop is the 0.15% of the prompt that holds the answer. Second, the textbook "optimal" eviction rule from computer caching (Belady's) does <em>worse</em> than the best online rules, because its assumptions do not fit attention. Third, a simple greedy oracle turns out to be essentially exactly optimal, so the gaps below are real gaps, not artefacts of a weak yardstick.</p>
      <p>Live results for question answering and the needle task are in. Live summarization is partly in and still running.</p>
    </div>
    <dl class="figures">
      <div><dt>Prompts</dt><dd class="ss-px ss-px--s">30</dd></div>
      <div><dt>Trace GB</dt><dd class="ss-px ss-px--s">6.5</dd></div>
      <div><dt>Policies</dt><dd class="ss-px ss-px--s">7</dd></div>
    </dl>
  </div>

  <header class="ss-chapter"><span class="ss-chapter__no">A</span><span class="ss-chapter__label">Background</span><h2>The words you need</h2></header>
  <dl class="glossary">
    <div><dt class="ss-label">Token</dt><dd>The unit a model reads and writes: a word, part of a word, a digit or a punctuation mark. A 12,000-token prompt is roughly a 9,000-word document.</dd></div>
    <div><dt class="ss-label">Attention</dt><dd>At every step the model decides how much to look at each earlier token. Those weights are non-negative and add up to 1 across all earlier tokens, so they behave like a budget of focus spread over the past.</dd></div>
    <div><dt class="ss-label">KV cache</dt><dd>For every token already read, the model stores two vectors per layer, a <em>key</em> (what this token offers) and a <em>value</em> (what it contributes). Storing them avoids recomputing the past at each new word. It grows linearly with length and is the main memory cost of long contexts.</dd></div>
    <div><dt class="ss-label">Prefill and decode</dt><dd><em>Prefill</em> reads the whole prompt in one pass and builds the cache. <em>Decode</em> then writes the answer one token at a time, each step reading from the cache.</dd></div>
    <div><dt class="ss-label">Budget and eviction</dt><dd>The <em>budget</em> is how many past tokens the cache may hold, here 5%, 10%, 25% or 50% of the prompt. <em>Eviction</em> is deleting a token's key and value to stay under it. In a KV cache an evicted token is gone for good; it cannot be fetched back later.</dd></div>
    <div><dt class="ss-label">Eviction policy (heuristic)</dt><dd>The rule that picks what to delete. An <em>online</em> policy only knows the past, which is all a real system ever knows.</dd></div>
    <div><dt class="ss-label">Oracle</dt><dd>An <em>offline</em> policy that is allowed to see the future: which tokens the model will look at later. It cannot be built in practice, but it shows the best any policy could do, so the gap to it measures how much room is left.</dd></div>
    <div><dt class="ss-label">Belady's MIN</dt><dd>The classic optimal rule for ordinary caches (1966): evict the item whose next use is furthest in the future. It is provably optimal when every hit is worth the same and a missed item can be re-fetched.</dd></div>
    <div><dt class="ss-label">Attention mass</dt><dd>Our offline score. At each decode step, add up the attention the full model pays to tokens a policy still holds. Keeping 0.90 of the oracle's mass means the policy kept 90% of what the oracle kept.</dd></div>
    <div><dt class="ss-label">Attention sinks</dt><dd>The first few tokens of a prompt soak up a large share of attention whatever they say, acting as a resting place for "nothing important here". Every sensible policy keeps them.</dd></div>
    <div><dt class="ss-label">KV head (GQA)</dt><dd>This model has 28 layers, each with 28 attention heads that share 4 sets of keys and values (grouped-query attention). We evict per layer and per shared set: 112 small caches, each with the same budget.</dd></div>
    <div><dt class="ss-label">Threshold τ</dt><dd>To use Belady's rule we must say when a token is "used". We call it a use when attention to it exceeds τ = 0.01, about 100 times what an even spread would give it.</dd></div>
  </dl>

  <header class="ss-chapter"><span class="ss-chapter__no">B</span><span class="ss-chapter__label">Method</span><h2>What we compared</h2></header>
  <div class="ss-table-wrap">
    <table class="ss-table">
      <thead><tr><th>Policy</th><th>Sees the future?</th><th>What it keeps</th></tr></thead>
      <tbody>
        <tr><td>Random</td><td>No</td><td>A random set. The floor any rule should beat.</td></tr>
        <tr><td>LRU</td><td>No</td><td>Tokens that were looked at (attention above τ) most recently. Least recently used goes first.</td></tr>
        <tr><td>StreamingLLM</td><td>No</td><td>The 4 sink tokens plus the most recent tokens. Ignores content entirely.</td></tr>
        <tr><td>H2O</td><td>No</td><td>Half the budget for "heavy hitters" (most total attention received so far), half for the most recent tokens.</td></tr>
        <tr><td>SnapKV</td><td>No</td><td>Tokens the last 32 queries looked at most, smoothed so neighbours of an important token survive too, plus the last 32 tokens.</td></tr>
        <tr><td>Belady oracle</td><td>Yes</td><td>Tokens whose next use above τ comes soonest.</td></tr>
        <tr><td>Weighted oracle</td><td>Yes</td><td>Tokens with the most attention still to come. This is the 1.0 every other number is measured against.</td></tr>
      </tbody>
    </table>
  </div>
  <div class="ss-prose"><p>Tasks: 10 synthetic needle prompts (a 7-digit "magic number" hidden at a known depth in 9–14k tokens of government reports), 10 government-report summaries and 10 multi-document questions from the LongBench benchmark. The model is Qwen2.5-7B-Instruct, 4-bit, on one RTX 3060. For each prompt we ran the model once with the full cache and recorded, at every step, how much every layer looked at every earlier token. Every policy was then replayed against those recordings.</p></div>

  <figure class="ss-plate">
    <header class="ss-plate__head ss-label"><span><b>PLATE 1</b> · HEURISTICS VS ORACLE · 10% BUDGET</span><span>OFFLINE SIM · n = {NS["needle"]} / {NS["summ"]} / {NS["mdqa"]}</span></header>
    <div class="ss-plate__body panels">{plate1()}</div>
    <figcaption class="ss-plate__cap">Plate 1. Share of the weighted oracle's attention mass each policy keeps when the cache may hold 10% of the prompt. Solid bars are online rules; the dashed outline is the Belady oracle, which sees the future and still trails SnapKV and H2O.</figcaption>
  </figure>

  <div class="ss-prose"><p><strong>The classic optimal rule loses to online heuristics.</strong> On needle prompts at 10% budget, Belady keeps 0.82 of the oracle's mass and online SnapKV keeps 0.95. The reason: per attention head only about 14 tokens ever cross τ during the answer, but the budget is about 1,150 tokens. Belady protects those 14 and fills the other 99% of the cache at random, throwing away the many tokens that each get a little attention and together carry about 30% of it. Belady is optimal only when every hit counts the same and misses can be re-fetched. Attention breaks both assumptions: it is weighted, and an evicted token never comes back.</p></div>

  <div class="ss-table-wrap">
    <table class="ss-table">
      <thead><tr><th>Task</th><th class="num">Belady τ = 0.001</th><th class="num">τ = 0.003</th><th class="num">τ = 0.01</th><th class="num">τ = 0.03</th></tr></thead>
      <tbody>{belsweep}</tbody>
    </table>
  </div>
  <div class="ss-prose"><p>Lowering the threshold helps Belady, because more tokens count as "used" and fewer slots are filled at random. At τ = 0.001 it reaches 0.89–0.94, close to the best heuristics, but still below the weighted oracle. The two oracles keep very different sets: at budgets of 25% or less only 3–16% of their kept tokens overlap (Jaccard index, the size of the overlap divided by the size of the union).</p>
  <p><strong>Is the weighted oracle really the best possible?</strong> It is greedy: at each step it drops the token with the least attention still to come, which is not guaranteed optimal. To check, we wrote the exact problem as a linear program (choose which tokens to hold at every step, never more than the budget, no re-admitting a dropped token) and solved it for {len(EX)} cases: 6 prompts, 6 heads each, budgets 5% and 25%, over the first 32 decode steps. Greedy reached on average {np.mean(EXR):.5f} of the true optimum, {min(EXR):.4f} in the worst case, and the solver's answer was a clean keep-or-drop choice every time, so it is the exact optimum and not a looser bound. For this objective, greedy is optimal in practice.</p></div>

  <figure class="ss-plate">
    <header class="ss-plate__head ss-label"><span><b>PLATE 2</b> · BEST HEURISTIC BY BUDGET</span><span>OFFLINE SIM</span></header>
    <div class="ss-plate__body">{plate2()}</div>
    <figcaption class="ss-plate__cap">Plate 2. Best of StreamingLLM, H2O and SnapKV at each budget, as a share of the weighted oracle. The gap grows a little as the budget shrinks on needle and QA; on summarization it is flat between 5% and 25%.</figcaption>
  </figure>

  <figure class="ss-plate">
    <header class="ss-plate__head ss-label"><span><b>PLATE 3</b> · WHERE NEEDLE ATTENTION GOES</span><span>MEDIAN OVER 112 HEADS · NEEDLE-00</span></header>
    <div class="ss-plate__body">{plate3()}</div>
    <figcaption class="ss-plate__cap">Plate 3. While writing the answer, the model puts about a third of its attention on 4 sink tokens and another 29% on the 64 tokens just before the answer, which hold the question. The hidden number is a few tokens inside the remaining 38%. A policy that keeps sinks and recent tokens already scores well on mass without keeping the number.</figcaption>
  </figure>

  <header class="ss-chapter"><span class="ss-chapter__no">C</span><span class="ss-chapter__label">Live runs</span><h2>Does the model still answer?</h2></header>
  <div class="ss-prose"><p>The numbers above are a simulation: replay a recording and count attention. Here each rule ran for real. The model read the prompt, the cache was cut to the budget, and the model wrote its answer while the rule evicted one token per new token. "Oracle, run live" replays the weighted oracle's evictions. Scores: needle is the share of the 10 prompts where the exact 7-digit number came back; question answering is token F1, the word overlap with the reference answer (0 to 1).</p></div>
  <div class="ss-table-wrap">
    <table class="ss-table">
      <thead><tr><th>Needle in a haystack</th><th class="num">5%</th><th class="num">10%</th><th class="num">25%</th><th class="num">50%</th></tr></thead>
      <tbody>{livetable("needle")}</tbody>
    </table>
  </div>
  <div class="ss-table-wrap">
    <table class="ss-table">
      <thead><tr><th>Multi-doc QA</th><th class="num">5%</th><th class="num">10%</th><th class="num">25%</th><th class="num">50%</th></tr></thead>
      <tbody>{livetable("mdqa")}</tbody>
    </table>
  </div>
  {('<div class="ss-table-wrap"><table class="ss-table"><thead><tr><th>Summarization (ROUGE-L) · ' + f'{live_full("summ")[1]} of 10 prompts so far' + '</th><th class="num">5%</th><th class="num">10%</th><th class="num">25%</th><th class="num">50%</th></tr></thead><tbody>' + livetable("summ") + '</tbody></table></div>') if live_full("summ")[1] else '<div class="ss-prose"><p>Summarization (10 prompts × 24 runs of 255 tokens each) is still waiting its turn on the shared GPU.</p></div>'}

  <div class="ss-note ss-note--rule"><span class="ss-note__tag">Main finding</span><div class="ss-note__body"><p><strong>Attention kept is not answers kept.</strong> At a 10% budget, LRU and H2O keep 0.88 and 0.84 of the oracle's attention on the needle task and find the number 0 times out of 10. SnapKV keeps 0.95 and finds it 10 out of 10, exactly like the oracle. A few points of attention mass separate total failure from perfect retrieval, because those few points sit on the answer.</p></div></div>

  <div class="ss-prose"><p>On question answering the picture is milder, because answers can often be pieced together from several places. H2O and SnapKV stay close to the full-cache score down to 5%; StreamingLLM, which keeps only the start and the end of the prompt, loses about half. With 10 prompts, differences of a few hundredths are within noise.</p></div>

  <figure class="ss-plate">
    <header class="ss-plate__head ss-label"><span><b>PLATE 4</b> · ATTENTION ON THE HIDDEN NUMBER THAT SURVIVES</span><span>NEEDLE · 10% BUDGET · n = 10</span></header>
    <div class="ss-plate__body">{plate4()}</div>
    <figcaption class="ss-plate__cap">Plate 4. Of the attention the full model pays to the 18 tokens of the hidden sentence while answering, the share each rule still holds. This, not total mass, lines up with the live results.</figcaption>
  </figure>
  <div class="ss-prose">
    <p><strong>Why the rules fail.</strong> The hidden sentence is 18 tokens, 0.15% of the prompt, yet it draws 10% of all attention while the model answers. That attention comes from a small set of <em>retrieval heads</em>, about 43 of the 112 caches, concentrated in the middle and late layers (14, 16, 19, 22). While the model is still reading the document, though, nothing points at a random number: the sentence is <em>cold</em>. H2O ranks tokens by attention gathered so far, which favours early tokens, so it keeps almost none of it ({FAIL["h2o"]["needle_mass"]/FAIL["full_needle_mass"]:.2f}). LRU keeps the sentence's words, which neighbouring words keep looking at, but only 43% of its digits, so the model copies a wrong number (6085932 for 6088743). StreamingLLM keeps only the start and end, and the needle is in the middle. SnapKV wins because its observation window is the last 32 tokens of the prompt, which <em>is the question</em>, and the question already looks at the number.</p>
    <p>This is a hypothesis with testable predictions, not yet tested: moving the question before the document should break SnapKV too, and repeating the question before compressing should rescue H2O and LRU.</p>
  </div>

  <header class="ss-chapter"><span class="ss-chapter__no">D</span><span class="ss-chapter__label">Predictions</span><h2>What we expected and where it stands</h2></header>
  <div class="ss-table-wrap">
    <table class="ss-table">
      <thead><tr><th>Prediction (written before any run)</th><th>Evidence so far, 10% budget</th><th>Status</th></tr></thead>
      <tbody>
        <tr><td>P1. Heuristics recover more than 90% of the oracle's mass on summarization.</td><td>Best is H2O at {agg("summ","0.1","h2o")[0]:.2f}; 0.90 at 5%, 0.94 at 50%. Right at the line.</td><td><span class="ss-pill ss-pill--mute">Unclear</span></td></tr>
        <tr><td>P2. Heuristics recover less than 60% on needle retrieval.</td><td>On mass, wrong: SnapKV keeps {agg("needle","0.1","snapkv")[0]:.2f}. Live, three of the four real heuristics (LRU, StreamingLLM, H2O) score 0/10, while SnapKV matches the oracle at 10/10. The prediction was about the wrong thing: the gap is not spread across heuristics, it is all or nothing per heuristic.</td><td><span class="ss-pill ss-pill--mute">Refuted on mass, split live</span></td></tr>
        <tr><td>P3. The gap widens as the budget shrinks.</td><td>On mass: yes for needle (0.98 → 0.94) and QA (0.98 → 0.91); flat for summarization. Live: yes for needle (most rules recover some prompts only at 25–50%) and for QA (StreamingLLM 0.43 → 0.29).</td><td><span class="ss-pill ss-pill--info">Partly supported</span></td></tr>
      </tbody>
    </table>
  </div>

  <header class="ss-chapter"><span class="ss-chapter__no">E</span><span class="ss-chapter__label">All numbers</span><h2>Share of oracle mass by budget</h2></header>
  <div class="ss-table-wrap">
    <table class="ss-table">
      <thead><tr><th>Task</th><th>Policy</th><th class="num">5%</th><th class="num">10%</th><th class="num">25%</th><th class="num">50%</th></tr></thead>
      <tbody>{numtable}</tbody>
    </table>
  </div>
  <div class="ss-prose"><p>For scale: even at a 5% budget, the oracle keeps most of the full cache's attention ({oracle_full}). Attention is concentrated, which is why heavy compression is possible at all.</p></div>

  <header class="ss-chapter"><span class="ss-chapter__no">F</span><span class="ss-chapter__label">Surprises</span><h2>Things that went differently</h2></header>
  <div class="ss-note ss-note--trap"><span class="ss-note__tag">Trap</span><div class="ss-note__body"><p><strong>Half-precision attention breaks this model.</strong> Computing attention scores in 16-bit floats, as the standard "eager" mode does, overflowed: some key vectors reach values of about 172, and the model produced strings of "!!!". Scores are now computed in 32-bit, and outputs match the library's reference path. Attention recordings made the other way on Qwen2 models can be silently wrong.</p></div></div>
  <div class="ss-note ss-note--trap"><span class="ss-note__tag">Trap</span><div class="ss-note__body"><p><strong>Attention mass is not accuracy.</strong> Most mass goes to sinks and recent tokens (Plate 3), so mass scores look high for any rule that keeps those. Confirmed live: 0.84–0.88 of oracle mass, 0 of 10 needles found.</p></div></div>
  <div class="ss-prose"><p><strong>Early layers do not care which rule you use.</strong> In layers 1, 5 and 6 every non-random rule keeps 0.90–0.98 of the oracle's attention, within a few hundredths of each other, because attention there is almost all sinks and recent tokens. The rules only separate in layers 14–27, where the retrieval heads live (needle, layer 14: StreamingLLM 0.50, SnapKV 0.94). That suggests giving later layers a bigger share of the cache than early ones. Not tested yet.</p></div>
  <div class="ss-prose">
    <p>Other caveats. Needle and QA answers are short (2–7 decode steps), so the oracle sees only a short future there. The first answer token always comes from the full-cache prefill pass, before any eviction, so on the needle task the first digit is free. SnapKV only compresses the prompt in its paper; to hold a fixed budget while writing we extended it to keep voting with the last 32 steps. No policy was tuned on these prompts; all settings are the papers' defaults.</p>
  </div>

  <header class="ss-chapter"><span class="ss-chapter__no">→</span><span class="ss-chapter__label">Next</span><h2>What runs next</h2></header>
  <div class="ss-table-wrap">
    <table class="ss-table">
      <thead><tr><th>Step</th><th>What it answers</th><th>Status</th></tr></thead>
      <tbody>
        <tr><td>Simulations, all 30 prompts</td><td>Attention kept by every rule at every budget</td><td><span class="ss-pill ss-pill--info">Done</span></td></tr>
        <tr><td>Exact-optimum check</td><td>Greedy oracle vs the true optimum</td><td><span class="ss-pill ss-pill--info">Done: within 0.07%</span></td></tr>
        <tr><td>Live accuracy, needle and QA</td><td>Does the model still answer when each rule runs for real?</td><td><span class="ss-pill ss-pill--info">Done</span></td></tr>
        <tr><td>Read failures</td><td>Which tokens the oracle kept that the rules dropped</td><td><span class="ss-pill ss-pill--info">Done (needle)</span></td></tr>
        <tr><td>Live accuracy, summarization</td><td>Does a 255-token summary survive eviction?</td><td><span class="ss-pill ss-pill--mute">Waiting for GPU</span></td></tr>
      </tbody>
    </table>
  </div>
  <div class="ss-note ss-note--verify"><span class="ss-note__tag">Open question</span><div class="ss-note__body">Answered for retrieval: total attention mass is the wrong yardstick. Open: should the oracle itself maximize attention on answer-bearing tokens in retrieval heads, and does any online rule do that without seeing the question first?</div></div>

  <footer class="ss-foot">
    {base[base.index('<div class="ss-lockup">'): base.index('<span class="ss-label">Report')]}<span class="ss-label">kvoracle · 2026-11-06</span>
  </footer>
</div>
</div>
'''
OUT.write_text(page)
print(OUT, len(page))
