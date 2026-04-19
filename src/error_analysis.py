"""Auto-bucketed error analysis.

The rubric says Analysis section determines high grades. We build an auto-bucketer
so that after every experiment cell we get:
  - % of errors in each linguistic bucket
  - examples per bucket for the paper appendix

Buckets we detect:
  negation   — presence of Urdu negators
  code_mix   — Roman/Latin characters mixed with Urdu script
  script     — text is Roman-Urdu only (no Urdu script at all)
  short      — <= 5 tokens (very little context)
  long       — >= 40 tokens (possible context dilution)
  sarcasm_like — exclamation + emoji-heavy, or contains common sarcasm markers
  refused    — model returned 'unknown' (instruction not followed)
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import re
import json
from collections import defaultdict

from .data import Example, LABELS


URDU_BLOCK = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN_BLOCK = re.compile(r"[A-Za-z]")
URDU_NEGATORS = ["نہیں", "نہ ", "نا ", "کبھی نہیں", "بالکل نہیں", "کوئی نہیں"]
# English translations users often code-mix
ROMAN_NEGATORS = [" nahi ", " nahin ", " mat ", " na "]
SARCASM_MARKERS_UR = ["واہ", "کیا بات ہے", "زبردست!", "شاباش"]  # often sarcastic in negative context


def tokenize_simple(text: str) -> List[str]:
    # Lightweight: split on whitespace + punctuation. Not perfect for Urdu but fine for bucketing.
    return re.findall(r"\S+", text)


def bucket_example(ex: Example, pred: str) -> List[str]:
    text = ex.text
    buckets = []
    has_urdu = bool(URDU_BLOCK.search(text))
    has_latin = bool(LATIN_BLOCK.search(text))

    if pred == "unknown":
        buckets.append("refused")

    # negation
    neg_hit = any(n in text for n in URDU_NEGATORS) or any(n in f" {text.lower()} " for n in ROMAN_NEGATORS)
    if neg_hit:
        buckets.append("negation")

    # script
    if has_urdu and has_latin:
        buckets.append("code_mix")
    elif not has_urdu and has_latin:
        buckets.append("roman_urdu_only")

    # length
    toks = tokenize_simple(text)
    if len(toks) <= 5:
        buckets.append("short")
    elif len(toks) >= 40:
        buckets.append("long")

    # sarcasm heuristic
    excl = text.count("!") + text.count("!")
    if excl >= 2 or any(m in text for m in SARCASM_MARKERS_UR):
        buckets.append("sarcasm_like")

    if not buckets:
        buckets.append("other")
    return buckets


def analyze(
    examples: List[Example],
    preds: List[str],
) -> Dict:
    """Return aggregate error statistics per bucket."""
    assert len(examples) == len(preds)
    total_errors = 0
    bucket_err = defaultdict(int)
    bucket_total = defaultdict(int)
    per_bucket_examples: Dict[str, List[Dict]] = defaultdict(list)

    for ex, pred in zip(examples, preds):
        buckets = bucket_example(ex, pred)
        correct = (pred == ex.label)
        for b in buckets:
            bucket_total[b] += 1
            if not correct:
                bucket_err[b] += 1
                if len(per_bucket_examples[b]) < 5:
                    per_bucket_examples[b].append({
                        "text": ex.text[:300],
                        "gold": ex.label,
                        "pred": pred,
                    })
        if not correct:
            total_errors += 1

    # Confusion pairs
    confusion_pairs = defaultdict(int)
    for ex, pred in zip(examples, preds):
        if pred != ex.label:
            confusion_pairs[f"{ex.label}->{pred}"] += 1

    return {
        "total_errors": total_errors,
        "total_n": len(examples),
        "error_rate": total_errors / max(1, len(examples)),
        "bucket_errors": dict(bucket_err),
        "bucket_totals": dict(bucket_total),
        "bucket_error_rates": {
            b: (bucket_err[b] / bucket_total[b]) for b in bucket_total if bucket_total[b] > 0
        },
        "confusion_pairs": dict(confusion_pairs),
        "sample_errors": {k: v for k, v in per_bucket_examples.items()},
    }
