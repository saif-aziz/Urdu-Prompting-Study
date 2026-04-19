"""Dataset loading with multi-source fallback.

Strategy:
1. Try each HF dataset candidate in order. Normalize label columns.
2. If all fail, load from local CSV (guaranteed to work — we ship a small seed file).
3. Split train/test if not pre-split.

Why this matters: dataset names on HF change, and code-due-tomorrow means we
can't afford a single dataset being unavailable to stall the whole pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import random
import pandas as pd

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


@dataclass
class Example:
    text: str
    label: str   # one of LABELS
    idx: int


def _normalize_label(raw) -> Optional[str]:
    """Coerce arbitrary dataset label conventions to {negative, neutral, positive}."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in LABELS:
        return s
    # numeric conventions
    mapping_num = {
        "0": "negative", "1": "neutral", "2": "positive",
        "-1": "negative", "1.0": "neutral", "2.0": "positive",
    }
    if s in mapping_num:
        return mapping_num[s]
    # text conventions
    mapping_text = {
        "neg": "negative", "pos": "positive", "neu": "neutral",
        "negative sentiment": "negative", "positive sentiment": "positive",
        "manfi": "negative", "musbat": "positive",   # roman-urdu common
    }
    return mapping_text.get(s)


def _load_hf(name: str, text_col: str, label_col: str) -> Optional[pd.DataFrame]:
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        ds = load_dataset(name)
    except Exception as e:
        print(f"[data] HF load failed for {name}: {e}")
        return None
    # Pick a split — prefer train
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split_name].to_pandas()
    # Fuzzy column resolution
    cols_lower = {c.lower(): c for c in df.columns}
    txt = cols_lower.get(text_col, None) or cols_lower.get("text") or cols_lower.get("sentence") or cols_lower.get("tweet")
    lab = cols_lower.get(label_col, None) or cols_lower.get("label") or cols_lower.get("sentiment") or cols_lower.get("class")
    if txt is None or lab is None:
        print(f"[data] {name}: could not resolve text/label columns (have {list(df.columns)})")
        return None
    out = pd.DataFrame({"text": df[txt].astype(str), "label_raw": df[lab]})
    out["label"] = out["label_raw"].apply(_normalize_label)
    out = out.dropna(subset=["label"])
    if len(out) < 50:
        return None
    print(f"[data] Loaded {name}: {len(out)} rows after normalization")
    return out[["text", "label"]].reset_index(drop=True)


def _load_local(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expected columns: text, label
    df["label"] = df["label"].apply(_normalize_label)
    df = df.dropna(subset=["label"])
    return df[["text", "label"]].reset_index(drop=True)


def load_dataset_robust(cfg) -> pd.DataFrame:
    for name in cfg.dataset.hf_candidates:
        df = _load_hf(name, cfg.dataset.text_column, cfg.dataset.label_column)
        if df is not None:
            return df
    print(f"[data] All HF candidates failed. Falling back to local CSV: {cfg.dataset.local_csv}")
    return _load_local(cfg.dataset.local_csv)


def split_train_test(df: pd.DataFrame, seed: int, test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)
    n_test = int(len(df) * test_frac)
    test_idx = sorted(idxs[:n_test])
    train_idx = sorted(idxs[n_test:])
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def prepare(cfg) -> Tuple[List[Example], List[Example]]:
    df = load_dataset_robust(cfg)
    train_df, test_df = split_train_test(df, cfg.seed)

    # Cap sizes
    if cfg.dataset.max_train and len(train_df) > cfg.dataset.max_train:
        train_df = train_df.sample(n=cfg.dataset.max_train, random_state=cfg.seed).reset_index(drop=True)
    if cfg.dataset.max_test and len(test_df) > cfg.dataset.max_test:
        # Stratified sample to keep class balance on test set
        pieces = []
        per = cfg.dataset.max_test // len(LABELS)
        for l in LABELS:
            sub = test_df[test_df["label"] == l]
            if len(sub) == 0:
                continue
            pieces.append(sub.sample(n=min(per, len(sub)), random_state=cfg.seed))
        test_df = pd.concat(pieces).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

    train = [Example(text=r["text"], label=r["label"], idx=i) for i, r in train_df.iterrows()]
    test = [Example(text=r["text"], label=r["label"], idx=i) for i, r in test_df.iterrows()]

    # Logging
    from collections import Counter
    print(f"[data] Train: {len(train)} | Test: {len(test)}")
    print(f"[data] Train class dist: {Counter(e.label for e in train)}")
    print(f"[data] Test  class dist: {Counter(e.label for e in test)}")
    return train, test


def select_few_shot(pool: List[Example], k: int, balanced: bool, seed: int) -> List[Example]:
    """Deterministic few-shot selection. If balanced, pick k/len(LABELS) per class."""
    if k == 0:
        return []
    rng = random.Random(seed)
    if balanced:
        per_class = max(1, k // len(LABELS))
        picked: List[Example] = []
        for l in LABELS:
            sub = [e for e in pool if e.label == l]
            rng.shuffle(sub)
            picked.extend(sub[:per_class])
        # Trim/pad to exactly k
        rng.shuffle(picked)
        if len(picked) > k:
            picked = picked[:k]
        elif len(picked) < k:
            remaining = [e for e in pool if e not in picked]
            rng.shuffle(remaining)
            picked.extend(remaining[: k - len(picked)])
        return picked
    else:
        copy = list(pool)
        rng.shuffle(copy)
        return copy[:k]
