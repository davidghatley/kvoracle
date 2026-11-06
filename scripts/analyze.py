"""Step 4: aggregate offline-sim and live results into markdown tables (docs/results.md) and charts (docs/figs/)."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = ["0.05", "0.1", "0.25", "0.5"]
TASKS = [("needle", "Needle"), ("summ", "Summarization"), ("mdqa", "Multi-doc QA")]
HEUR = ["random", "lru", "streaming", "h2o", "snapkv"]
REAL = ["streaming", "h2o", "snapkv"]  # "current heuristics" for P1-P3 (see RESEARCH.md operationalization)
OFF = HEUR + ["belady@0.01", "oracle_w", "ceiling"]
LIVE = HEUR + ["oracle_replay"]
COL = dict(random="#52514e", lru="#e87ba4", streaming="#eb6834", h2o="#1baf7a", snapkv="#2a78d6",
           **{"belady@0.01": "#6250d6", "oracle_w": "#0b0b0b", "oracle_replay": "#0b0b0b", "ceiling": "#a3a29b", "full": "#a3a29b"})
LBL = {"belady@0.01": "Belady (τ=0.01)", "oracle_w": "Weighted oracle", "oracle_replay": "Oracle replay",
       "ceiling": "Per-step ceiling", "streaming": "StreamingLLM", "h2o": "H2O", "snapkv": "SnapKV",
       "lru": "LRU", "random": "Random", "full": "Full cache"}

sims = {p.stem: json.load(open(p)) for p in (ROOT / "data/sim").glob("*.json") if p.stem not in ("exact_check", "failures")}
lives = {p.stem: json.load(open(p)) for p in (ROOT / "data/live").glob("*.json")} if (ROOT / "data/live").exists() else {}


def ratio(r, k, den="oracle_w", heads=slice(None)):
    return sum(r[k][heads]) / sum(r[den][heads])


def off_table(den="oracle_w"):
    """mean over prompts of (policy mass / den mass), summed over heads and steps within a prompt."""
    out = {}
    for task, _ in TASKS:
        ps = [s for s in sims.values() if s["meta"]["task"] == task]
        for b in BUDGETS:
            for k in OFF + ["full"]:
                v = [ratio(s[b], k, den) for s in ps]
                out[task, b, k] = (np.mean(v), np.std(v), len(v)) if v else (np.nan, np.nan, 0)
    return out


def live_table():
    out = {}
    tasks = {json.loads(l)["id"]: json.loads(l)["task"] for l in open(ROOT / "data/tasks.jsonl")}
    for task, _ in TASKS:
        ps = [r for pid, r in lives.items() if tasks[pid] == task]
        out[task, "full"] = (np.mean([r["full"]["score"] for r in ps]) if ps else np.nan, len(ps))
        for b in BUDGETS:
            for k in LIVE:
                v = [r[f"{float(b)}|{k}"]["score"] for r in ps if f"{float(b)}|{k}" in r]
                a = [r[f"{float(b)}|{k}"]["agree"] for r in ps if f"{float(b)}|{k}" in r]
                out[task, b, k] = (np.mean(v) if v else np.nan, np.mean(a) if a else np.nan, len(v))
    return out


def md():
    o = off_table()
    lines = ["### (a) Attention mass retained, as a fraction of the weighted oracle (offline sim, mean ± sd over 10 prompts)", ""]
    for task, name in TASKS:
        lines += [f"**{name}**", "", "| policy | " + " | ".join(f"{float(b):.0%}" for b in BUDGETS) + " |",
                  "|---|" + "---|" * len(BUDGETS)]
        for k in OFF:
            lines.append(f"| {LBL.get(k, k)} | " + " | ".join(f"{o[task, b, k][0]:.3f} ± {o[task, b, k][1]:.3f}" for b in BUDGETS) + " |")
        lines.append("| *oracle mass / full-cache mass* | " + " | ".join(f"{1 / o[task, b, 'full'][0]:.3f}" for b in BUDGETS) + " |")
        lines.append("")
    if lives:
        lv = live_table()
        lines += ["### (b) Live task score (needle: exact-match rate; summarization: ROUGE-L; QA: token F1)", "",
                  "Cell = score (ROUGE-L agreement with the full-cache output in parentheses).", ""]
        for task, name in TASKS:
            lines += [f"**{name}** (full cache: {lv[task, 'full'][0]:.3f}, n={lv[task, 'full'][1]})", "",
                      "| policy | " + " | ".join(f"{float(b):.0%}" for b in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
            for k in LIVE:
                lines.append(f"| {LBL.get(k, k)} | " + " | ".join(f"{lv[task, b, k][0]:.3f} ({lv[task, b, k][1]:.2f})" for b in BUDGETS) + " |")
            lines.append("")
    # oracle divergence + tau sensitivity
    lines += ["### Oracle variants: Belady vs weighted greedy", "",
              "| task | budget | Belady τ=0.001 | τ=0.003 | τ=0.01 | τ=0.03 | keep-set Jaccard (τ=0.01) |", "|---|---|---|---|---|---|---|"]
    for task, name in TASKS:
        ps = [s for s in sims.values() if s["meta"]["task"] == task]
        for b in BUDGETS:
            cells = [f"{np.mean([ratio(s[b], f'belady@{x}') for s in ps]):.3f}" for x in [0.001, 0.003, 0.01, 0.03]]
            jac = np.mean([np.mean(s[b]["jaccard_belady_w"]) for s in ps])
            lines.append(f"| {name} | {float(b):.0%} | " + " | ".join(cells) + f" | {jac:.3f} |")
    lines.append("")
    # predictions
    lines += ["### Prediction check numbers: best of StreamingLLM / H2O / SnapKV, mass / weighted oracle", "",
              "| task | " + " | ".join(f"{float(b):.0%}" for b in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
    for task, name in TASKS:
        lines.append(f"| {name} | " + " | ".join(
            f"{max(o[task, b, k][0] for k in REAL):.3f} ({LBL[max(REAL, key=lambda k: o[task, b, k][0])]})" for b in BUDGETS) + " |")
    lines.append("")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/results.md").write_text("\n".join(lines))
    return o


def per_layer(b="0.1"):
    """[task][policy] -> 28 per-layer ratios to oracle (sum over the 4 kv heads and prompts)."""
    out = {}
    for task, _ in TASKS:
        ps = [s for s in sims.values() if s["meta"]["task"] == task]
        out[task] = {k: [np.mean([ratio(s[b], k, heads=slice(4 * l, 4 * l + 4)) for s in ps]) for l in range(28)] for k in HEUR + ["belady@0.01"]}
    return out


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.tick_params(colors="#52514e", labelsize=9)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#a3a29b")


def charts(o):
    figs = ROOT / "docs/figs"
    figs.mkdir(parents=True, exist_ok=True)
    x = [float(b) * 100 for b in BUDGETS]
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11})
    # fig 1: mass ratio vs budget
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (task, name) in zip(axs, TASKS):
        for k in HEUR + ["belady@0.01", "ceiling"]:
            y = [o[task, b, k][0] for b in BUDGETS]
            ax.plot(x, y, "o-", color=COL[k], lw=2, ms=6, label=LBL[k], ls="--" if k == "ceiling" else "-")
        ax.axhline(1, color="#0b0b0b", lw=1)
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v:g}%" for v in x])
        ax.set_title(name, loc="left", color="#0b0b0b"); ax.set_xlabel("cache budget (% of prompt)", color="#52514e")
        style(ax)
    axs[0].set_ylabel("attention mass / weighted oracle", color="#52514e")
    axs[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Retained attention mass relative to the weighted oracle (line at 1.0)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(); fig.savefig(figs / "mass_vs_budget.png", dpi=150); plt.close(fig)
    # fig 2: live scores
    if lives:
        lv = live_table()
        fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
        for ax, (task, name) in zip(axs, TASKS):
            if not lv[task, "full"][1]:
                ax.set_title(f"{name} (pending)", loc="left"); ax.axis("off"); continue
            for k in LIVE:
                ax.plot(x, [lv[task, b, k][0] for b in BUDGETS], "o-", color=COL[k], lw=2, ms=6, label=LBL[k])
            ax.axhline(lv[task, "full"][0], color="#a3a29b", lw=1.5, ls="--", label="Full cache")
            ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v:g}%" for v in x])
            ax.set_title(name, loc="left"); ax.set_xlabel("cache budget (% of prompt)", color="#52514e"); style(ax)
        axs[0].set_ylabel("task score", color="#52514e")
        axs[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Live task score under eviction", x=0.01, ha="left", fontsize=12)
        fig.tight_layout(); fig.savefig(figs / "live_vs_budget.png", dpi=150); plt.close(fig)
    # fig 3: per-layer
    pl = per_layer()
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (task, name) in zip(axs, TASKS):
        for k in HEUR + ["belady@0.01"]:
            ax.plot(range(28), pl[task][k], "-", color=COL[k], lw=2, label=LBL[k])
        ax.set_title(name, loc="left"); ax.set_xlabel("layer", color="#52514e"); style(ax)
    axs[0].set_ylabel("mass / weighted oracle (10% budget)", color="#52514e")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "per_layer_10pct.png", dpi=150); plt.close(fig)
    return pl


if __name__ == "__main__":
    o = md()
    pl = charts(o)
    print((ROOT / "docs/results.md").read_text())
    print("per-layer spread (max-min over heuristics) at 10%:")
    for task, _ in TASKS:
        sp = [max(pl[task][k][l] for k in REAL) - min(pl[task][k][l] for k in REAL) for l in range(28)]
        print(task, " ".join(f"{v:.2f}" for v in sp))
