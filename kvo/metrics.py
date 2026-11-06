"""Task scoring. Needle: exact number present. Multi-doc QA: LongBench token F1 (max over answers).
Summarization: ROUGE-L F (rouge_score, stemmed). Also ROUGE-L agreement with the full-cache output."""
import re
import string
from collections import Counter

from rouge_score import rouge_scorer

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    common = Counter(p) & Counter(g)
    k = sum(common.values())
    if k == 0:
        return 0.0
    pr, rc = k / len(p), k / len(g)
    return 2 * pr * rc / (pr + rc)


def rouge_l(pred, ref):
    return _rouge.score(ref, pred)["rougeL"].fmeasure


def score(task, pred, answers):
    if task == "needle":
        return float(any(a in pred for a in answers))
    if task == "mdqa":
        return max(qa_f1(pred, a) for a in answers)
    return max(rouge_l(pred, a) for a in answers)
