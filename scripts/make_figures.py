"""Generate the paper's figures and LaTeX tables from results/metrics.csv.

Outputs:
  paper/figures/heatmap_en_vs_ur.png    — the money figure
  paper/figures/per_bucket_errors.png   — error analysis visualisation
  paper/figures/confusion_matrices.png  — grid of confusion matrices
  results/main_results_table.tex        — IEEE-ready table to \\input{} in paper
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


RESULTS = Path("results")
FIGS = Path("paper/figures")
FIGS.mkdir(parents=True, exist_ok=True)


def load():
    m = pd.read_csv(RESULTS / "metrics.csv")
    with (RESULTS / "error_analysis.json").open(encoding="utf-8") as f:
        errs = json.load(f)
    with (RESULTS / "all_predictions.json").open(encoding="utf-8") as f:
        preds = json.load(f)
    with (RESULTS / "y_true.json").open(encoding="utf-8") as f:
        y_true = json.load(f)
    return m, errs, preds, y_true


def heatmap_en_vs_ur(metrics: pd.DataFrame):
    """Heatmap: rows = models, cols = prompt variants, cells = macro-F1.
    Separately draw en-vs-ur difference bars."""
    df = metrics[metrics["prompt"] != "-"].copy()
    pivot = df.pivot_table(index="model", columns="prompt", values="macro_f1")
    pivot = pivot.reindex(columns=["zs_en", "zs_ur", "fs_en", "fs_ur", "cot_en", "cot_ur"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1.2]})

    # Left: heatmap
    ax = axes[0]
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.6, pivot.values.max()))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.4 else "black", fontsize=9)
    ax.set_title("Macro-F1 across models and prompt variants")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Macro-F1")

    # Right: en-vs-ur deltas per model for each setting
    ax = axes[1]
    settings = ["zs", "fs", "cot"]
    xs = np.arange(len(pivot.index))
    width = 0.27
    for i, s in enumerate(settings):
        en_col, ur_col = f"{s}_en", f"{s}_ur"
        if en_col in pivot.columns and ur_col in pivot.columns:
            delta = pivot[en_col] - pivot[ur_col]
            ax.bar(xs + (i - 1) * width, delta.values, width, label=s)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel("F1(en) − F1(ur)")
    ax.set_title("English-prompt advantage")
    ax.legend(title="Setting", frameon=False)

    plt.tight_layout()
    plt.savefig(FIGS / "heatmap_en_vs_ur.png", dpi=150)
    plt.close()
    print(f"[fig] wrote {FIGS / 'heatmap_en_vs_ur.png'}")


def per_bucket_errors(errs: dict):
    """Bar chart: error rate per linguistic bucket, averaged over models, for en vs ur."""
    buckets_of_interest = ["negation", "code_mix", "roman_urdu_only",
                           "short", "long", "sarcasm_like", "refused"]

    def mean_rate_for_lang(suffix: str) -> dict:
        rows = {b: [] for b in buckets_of_interest}
        for k, v in errs.items():
            if not k.endswith(f"|{suffix}"):
                # We filter by suffix like 'zs_en'
                pass
            parts = k.split("|")
            if len(parts) < 2:
                continue
            if not parts[1].endswith(suffix):
                continue
            for b in buckets_of_interest:
                if b in v.get("bucket_error_rates", {}):
                    rows[b].append(v["bucket_error_rates"][b])
        return {b: (np.mean(v) if v else np.nan) for b, v in rows.items()}

    en = mean_rate_for_lang("_en")
    ur = mean_rate_for_lang("_ur")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(buckets_of_interest))
    ax.bar(xs - 0.2, [en[b] for b in buckets_of_interest], 0.4, label="EN prompts")
    ax.bar(xs + 0.2, [ur[b] for b in buckets_of_interest], 0.4, label="UR prompts")
    ax.set_xticks(xs)
    ax.set_xticklabels(buckets_of_interest, rotation=25, ha="right")
    ax.set_ylabel("Error rate in bucket (avg over models)")
    ax.set_title("Where do EN and UR prompts fail?")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGS / "per_bucket_errors.png", dpi=150)
    plt.close()
    print(f"[fig] wrote {FIGS / 'per_bucket_errors.png'}")


def confusion_grid(preds: dict, y_true: list):
    """Small grid of confusion matrices for every cell."""
    from sklearn.metrics import confusion_matrix
    keys = [k for k in preds.keys() if "|" in k]
    if not keys:
        return
    n = len(keys)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    labels = ["negative", "neutral", "positive"]
    for i, k in enumerate(keys):
        ax = axes[i]
        yp = [p if p in labels else "neutral" for p in preds[k]]
        cm = confusion_matrix(y_true, yp, labels=labels)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(k.replace("|", " / "), fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels(["neg", "neu", "pos"], fontsize=7)
        ax.set_yticks(range(3)); ax.set_yticklabels(["neg", "neu", "pos"], fontsize=7)
        for x in range(3):
            for y in range(3):
                ax.text(y, x, cm[x, y], ha="center", va="center",
                        color="white" if cm[x, y] > cm.max() / 2 else "black", fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "confusion_matrices.png", dpi=130)
    plt.close()
    print(f"[fig] wrote {FIGS / 'confusion_matrices.png'}")


def latex_table(metrics: pd.DataFrame):
    """Write an IEEE-style booktabs table of main results."""
    df = metrics.copy()
    df["key"] = df["model"] + " / " + df["prompt"].astype(str)
    df = df.sort_values(["model", "prompt"])
    lines = [
        "% Auto-generated by scripts/make_figures.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Main results. Accuracy and macro-F1 on Urdu sentiment test set. "
        "Best LLM cell per model is \\textbf{bold}.}",
        "\\label{tab:main}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Model & Prompt & Acc. & Macro-F1 & Unk.\\% \\\\",
        "\\midrule",
    ]
    # Determine bold per-model best macro_f1 (over LLM rows only)
    llm = df[df["prompt"] != "-"].copy()
    best_idx = llm.groupby("model")["macro_f1"].idxmax().to_dict()
    best_set = set(best_idx.values())

    for _, r in df.iterrows():
        mf1 = f"{r['macro_f1']:.3f}"
        acc = f"{r['accuracy']:.3f}"
        if r.name in best_set:
            mf1 = f"\\textbf{{{mf1}}}"
            acc = f"\\textbf{{{acc}}}"
        unk = f"{100 * r['unknown_rate']:.1f}"
        lines.append(f"{r['model']} & {r['prompt']} & {acc} & {mf1} & {unk} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    path = RESULTS / "main_results_table.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[tex] wrote {path}")


def main():
    metrics, errs, preds, y_true = load()
    heatmap_en_vs_ur(metrics)
    per_bucket_errors(errs)
    confusion_grid(preds, y_true)
    latex_table(metrics)


if __name__ == "__main__":
    main()
