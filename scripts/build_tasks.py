"""Build the evaluation task set (10 per task) plus one held-out dev prompt per task.

Selection is deterministic and blind to model outputs: LongBench rows in file order whose
chat-formatted prompt is 8k-16k tokens; needles are synthetic with fixed seed.
Output: data/tasks.jsonl (eval) and data/tasks_dev.jsonl (dev, never used for reported numbers).
"""
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
LB = ROOT / "data/longbench/data"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
LO, HI = 8000, 16000

# LongBench prompt templates (dataset2prompt.json) and max_gen; summaries capped at 256 (see RESEARCH.md D4)
TEMPLATES = {
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
}
MAX_NEW = {"summ": 256, "mdqa": 32, "needle": 24}

tok = AutoTokenizer.from_pretrained(MODEL)


def chat(user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def ntok(s):
    return len(tok(s, add_special_tokens=False).input_ids)


def longbench(name, task, n=11):
    out = []
    for line in open(LB / f"{name}.jsonl"):
        r = json.loads(line)
        p = chat(TEMPLATES[name].format(**r))
        L = ntok(p)
        if LO <= L <= HI:
            out.append(dict(task=task, id=f"{task}-{r['_id'][:8]}", prompt=p, n_prompt=L,
                            answers=r["answers"], max_new=MAX_NEW[task], source=name))
        if len(out) == n:
            return out
    raise RuntimeError(name)


def needles(n=11, seed=0):
    """Haystack = concatenated gov_report rows outside the 8-16k window (so never the summarization docs)."""
    rng = random.Random(seed)
    hay = [json.loads(l)["context"] for l in open(LB / "gov_report.jsonl")]
    hay = [h for h in hay if ntok(chat(TEMPLATES["gov_report"].format(context=h))) < LO]
    cities = ["Lisbon", "Nairobi", "Osaka", "Quito", "Tallinn", "Hobart", "Accra", "Bergen", "Cusco", "Da Nang", "Fresno"]
    out = []
    for i in range(n):
        target = rng.randint(8500, 15000)
        depth = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.35][i]
        code = str(rng.randint(1000000, 9999999))
        needle = f"\nThe special magic number for {cities[i]} is {code}.\n"
        q = f"What is the special magic number for {cities[i]}? Answer with the number only."
        docs, j = [], rng.randrange(len(hay))
        body = ""
        while ntok(body) < target - 300:
            body += hay[j % len(hay)] + "\n\n"
            j += 1
        ids = tok(body, add_special_tokens=False).input_ids[: target - 300]
        cut = int(len(ids) * depth)
        ctx = tok.decode(ids[:cut]) + needle + tok.decode(ids[cut:])
        p = chat(f"Below is a long document. A special magic number is hidden in it.\n\n{ctx}\n\n{q}")
        out.append(dict(task="needle", id=f"needle-{i:02d}", prompt=p, n_prompt=ntok(p), answers=[code],
                        max_new=MAX_NEW["needle"], source="synthetic", depth=depth, needle=needle.strip()))
    return out


if __name__ == "__main__":
    allt = needles() + longbench("gov_report", "summ") + longbench("hotpotqa", "mdqa")
    by = {}
    for t in allt:
        by.setdefault(t["task"], []).append(t)
    ev, dev = [], []
    for task, ts in by.items():
        ev += ts[:10]
        dev += ts[10:]
    with open(ROOT / "data/tasks.jsonl", "w") as f:
        for t in ev:
            f.write(json.dumps(t) + "\n")
    with open(ROOT / "data/tasks_dev.jsonl", "w") as f:
        for t in dev:
            f.write(json.dumps(t) + "\n")
    for t in ev + dev:
        print(t["id"], t["task"], t["n_prompt"], t.get("depth", ""))
