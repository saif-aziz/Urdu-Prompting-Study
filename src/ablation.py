"""Shot-count ablation: for the single best-performing (model, en-or-ur) cell,
sweep k in {0, 1, 3, 5} in both languages and plot the curves.

This adds the 'Ablation Study' the rubric explicitly requires.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_config
from .data import prepare, select_few_shot
from .prompts import build_prompt, parse_label
from .inference import load_model_and_tokenizer, generate_batch, unload
from .metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", required=True, help="short name from config")
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 1, 3, 5])
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_test:
        cfg.dataset.max_test = args.max_test

    model_cfg = next((m for m in cfg.models if m.short == args.model), None)
    if model_cfg is None:
        raise SystemExit(f"Model {args.model} not found in config")

    train, test = prepare(cfg)
    y_true = [e.label for e in test]

    print(f"[ablation] loading {model_cfg.name}")
    model, tok = load_model_and_tokenizer(model_cfg)

    records = []
    for lang in ("en", "ur"):
        for k in args.shots:
            shots = select_few_shot(train, k, cfg.few_shot.balanced, cfg.seed)
            if k == 0:
                variant = f"zs_{lang}"
            else:
                variant = f"fs_{lang}"
            prompts = [build_prompt(variant, ex.text, shots) for ex in test]
            raws = generate_batch(
                model, tok, prompts,
                use_chat_template=model_cfg.chat_template,
                max_new_tokens=cfg.generation.max_new_tokens,
                do_sample=False, temperature=1.0,
                batch_size=cfg.generation.batch_size,
            )
            preds = [parse_label(r, variant) for r in raws]
            m = compute_metrics(y_true, preds)
            print(f"  lang={lang}  k={k}  acc={m['accuracy']:.3f}  f1={m['macro_f1']:.3f}")
            records.append({"lang": lang, "k": k, "acc": m["accuracy"], "f1": m["macro_f1"]})

    unload(model)

    out_dir = Path("results")
    df = pd.DataFrame(records)
    df.to_csv(out_dir / f"ablation_shots_{args.model}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    for lang in ("en", "ur"):
        sub = df[df["lang"] == lang].sort_values("k")
        ax.plot(sub["k"], sub["f1"], marker="o", label=f"{lang.upper()} prompts")
    ax.set_xlabel("Number of few-shot examples (k)")
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"Shot-count ablation — {args.model}")
    ax.legend(frameon=False)
    fig_path = Path("paper/figures") / f"ablation_shots_{args.model}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[ablation] wrote {fig_path}")


if __name__ == "__main__":
    main()
