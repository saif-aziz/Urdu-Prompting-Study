"""Prompt templates for the six variants.

Design decisions:
- Keep each pair (en/ur) as close in semantic content as possible so any
  difference is attributable to the prompting language, not the task framing.
- Output format is always a single-word label to make parsing robust.
- CoT variants ask for reasoning THEN a final labeled line (we parse the line).

LABEL vocabulary the model must emit:
  English prompts expect: negative / neutral / positive
  Urdu prompts expect:    منفی / غیر جانبدار / مثبت  (mapped back to English)
"""
from __future__ import annotations
from typing import List
from .data import Example, LABELS


# Urdu label tokens. We accept common variants on parse.
UR_LABEL = {
    "negative": "منفی",
    "neutral":  "غیر جانبدار",
    "positive": "مثبت",
}
UR_LABEL_VARIANTS = {
    "negative": ["منفی", "منفى", "نیگیٹو"],
    "neutral":  ["غیر جانبدار", "غیرجانبدار", "نیوٹرل", "معتدل"],
    "positive": ["مثبت", "پازیٹو"],
}


# ---------------- English templates ----------------
EN_SYSTEM = (
    "You are a precise sentiment classifier for Urdu text. "
    "Respond with exactly one word from: negative, neutral, positive. "
    "Do not add punctuation or explanation."
)
EN_INSTRUCTION = "Classify the sentiment of the following Urdu text. Answer with one word: negative, neutral, or positive."
EN_COT_INSTRUCTION = (
    "Classify the sentiment of the following Urdu text. "
    "First briefly reason step by step in English, then on a new line write 'Answer: <label>' "
    "where <label> is one of: negative, neutral, positive."
)


# ---------------- Urdu templates ----------------
UR_SYSTEM = (
    "آپ ایک درست اردو جذبات کی درجہ بندی کرنے والے ہیں۔ "
    "صرف ایک لفظ سے جواب دیں: منفی، غیر جانبدار، یا مثبت۔ "
    "کوئی وضاحت یا علامات شامل نہ کریں۔"
)
UR_INSTRUCTION = (
    "درج ذیل اردو متن کے جذبات کی درجہ بندی کریں۔ "
    "صرف ایک لفظ سے جواب دیں: منفی، غیر جانبدار، یا مثبت۔"
)
UR_COT_INSTRUCTION = (
    "درج ذیل اردو متن کے جذبات کی درجہ بندی کریں۔ "
    "پہلے مختصر قدم بہ قدم اردو میں سوچیں، پھر نئی سطر پر لکھیں 'جواب: <لیبل>' "
    "جہاں <لیبل> ان میں سے ایک ہو: منفی، غیر جانبدار، مثبت۔"
)


def _format_shots_en(shots: List[Example]) -> str:
    if not shots:
        return ""
    lines = []
    for s in shots:
        lines.append(f"Text: {s.text}\nAnswer: {s.label}")
    return "\n\n".join(lines) + "\n\n"


def _format_shots_ur(shots: List[Example]) -> str:
    if not shots:
        return ""
    lines = []
    for s in shots:
        lines.append(f"متن: {s.text}\nجواب: {UR_LABEL[s.label]}")
    return "\n\n".join(lines) + "\n\n"


def build_prompt(variant: str, text: str, shots: List[Example]) -> dict:
    """Return a dict with 'system' and 'user' strings for the chat template."""
    if variant == "zs_en":
        return {
            "system": EN_SYSTEM,
            "user": f"{EN_INSTRUCTION}\n\nText: {text}\nAnswer:",
        }
    if variant == "zs_ur":
        return {
            "system": UR_SYSTEM,
            "user": f"{UR_INSTRUCTION}\n\nمتن: {text}\nجواب:",
        }
    if variant == "fs_en":
        shots_block = _format_shots_en(shots)
        return {
            "system": EN_SYSTEM,
            "user": f"{EN_INSTRUCTION}\n\n{shots_block}Text: {text}\nAnswer:",
        }
    if variant == "fs_ur":
        shots_block = _format_shots_ur(shots)
        return {
            "system": UR_SYSTEM,
            "user": f"{UR_INSTRUCTION}\n\n{shots_block}متن: {text}\nجواب:",
        }
    if variant == "cot_en":
        return {
            "system": EN_SYSTEM,
            "user": f"{EN_COT_INSTRUCTION}\n\nText: {text}",
        }
    if variant == "cot_ur":
        return {
            "system": UR_SYSTEM,
            "user": f"{UR_COT_INSTRUCTION}\n\nمتن: {text}",
        }
    raise ValueError(f"Unknown prompt variant: {variant}")


def parse_label(raw_output: str, variant: str) -> str:
    """Parse model output to one of LABELS. Returns 'unknown' if unparseable."""
    if raw_output is None:
        return "unknown"
    text = raw_output.strip().lower()

    # CoT variants: look for 'answer:' or 'جواب:' on last populated line
    if variant.startswith("cot_"):
        lines = [ln.strip() for ln in raw_output.split("\n") if ln.strip()]
        for ln in reversed(lines):
            low = ln.lower()
            if low.startswith("answer:") or "answer:" in low:
                text = low.split("answer:", 1)[1].strip()
                break
            if ln.startswith("جواب") or "جواب:" in ln:
                text = ln.split("جواب", 1)[-1].lstrip(":").strip().lower()
                # Map any Urdu label token in this fragment
                for en, variants in UR_LABEL_VARIANTS.items():
                    if any(v in ln for v in variants):
                        return en
                break

    # Direct English match
    for l in LABELS:
        if l in text:
            # Avoid false match where "negative" appears inside "non-negative"
            if l == "negative" and "non-negative" in text:
                continue
            return l

    # Direct Urdu match (full prompt or CoT body)
    for en, variants in UR_LABEL_VARIANTS.items():
        if any(v in raw_output for v in variants):
            return en

    return "unknown"
