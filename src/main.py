"""End-to-end experiment runner.

Usage:
    python -m src.main --config configs/default.yaml
    python -m src.main --config configs/default.yaml --max-test 50    # smoke test
    python -m src.main --config configs/default.yaml --skip-xlmr      # skip supervised baseline
    python -m src.main --config configs/default.yaml --only qwen05    # only one model

Outputs land in results/:
  predictions/{short}_{prompt}.jsonl
  metrics.csv
  error_analysis.json
  summary.json
"""
from __future__ import annotations
import argparse
import json
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

from .config import load_config
from .data import prepare, select_few_shot, LABELS
from .prompts import build_prompt, parse_label
from .inference import load_model_and_tokenizer, generate_batch, unload
from .metrics import compute_metrics, mcnemar_test, majority_class_baseline
from .error_analysis import analyze


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_one_cell(model, tokenizer, model_cfg, variant, test, shots, gen_cfg):
    prompts = [build_prompt(variant, ex.text, shots) for ex in test]
    raw_outputs = generate_batch(
        model, tokenizer, prompts,
        use_chat_template=model_cfg.chat_template,
        max_new_tokens=gen_cfg.max_new_tokens,
        do_sample=gen_cfg.do_sample,
        temperature=gen_cfg.temperature,
        batch_size=gen_cfg.batch_size,
    )
    preds = [parse_label(out, variant) for out in raw_outputs]
    return preds, raw_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Override max_test for quick runs")
    parser.add_argument("--skip-xlmr", action="store_true")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only the model with this 'short' name")
    parser.add_argument("--only-variant", type=str, default=None,
                        help="Run only this prompt variant")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_test is not None:
        cfg.dataset.max_test = args.max_test
    set_seed(cfg.seed)

    out_dir = Path(cfg.output_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "error_analysis").mkdir(parents=True, exist_ok=True)

    # ---- Data
    train, test = prepare(cfg)
    y_true = [e.label for e in test]
    few_shot_pool = select_few_shot(train, cfg.few_shot.k, cfg.few_shot.balanced, cfg.seed)
    print(f"[main] Few-shot pool (k={cfg.few_shot.k}): "
          f"{[(s.label, s.text[:40]) for s in few_shot_pool]}")

    all_metrics = []
    all_errors = {}
    all_preds = {}

    # ---- Majority-class baseline
    y_train = [e.label for e in train]
    maj = majority_class_baseline(y_train, y_true)
    print(f"[baseline/majority] acc={maj['accuracy']:.3f} f1={maj['macro_f1']:.3f}  label={maj['_label_used']}")
    all_preds["majority"] = maj["_predictions"]
    all_metrics.append({
        "model": "majority", "prompt": "-",
        "accuracy": maj["accuracy"], "macro_f1": maj["macro_f1"],
        "f1_negative": maj["f1_negative"], "f1_neutral": maj["f1_neutral"],
        "f1_positive": maj["f1_positive"],
        "unknown_rate": maj["unknown_rate"], "n": maj["n"],
    })

    # ---- XLM-R fine-tuned baseline
    if not args.skip_xlmr:
        try:
            from .baseline_finetune import train_and_eval_xlmr
            xlmr_result = train_and_eval_xlmr(cfg, train, test)
            xlmr_preds = xlmr_result["predictions"]
            m = compute_metrics(y_true, xlmr_preds)
            print(f"[baseline/xlmr] acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f}")
            all_preds["xlmr"] = xlmr_preds
            all_metrics.append({
                "model": "xlm-roberta-base", "prompt": "-",
                "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                "f1_negative": m["f1_negative"], "f1_neutral": m["f1_neutral"],
                "f1_positive": m["f1_positive"],
                "unknown_rate": m["unknown_rate"], "n": m["n"],
            })
            all_errors["xlm-roberta-base|-"] = analyze(test, xlmr_preds)
        except Exception as e:
            print(f"[baseline/xlmr] skipped due to error: {e}")

    # ---- Prompt x Model grid
    for model_cfg in cfg.models:
        if args.only and model_cfg.short != args.only:
            continue
        print(f"\n==== Loading {model_cfg.name} ====")
        model, tokenizer = load_model_and_tokenizer(model_cfg)

        for variant in cfg.prompts:
            if args.only_variant and variant != args.only_variant:
                continue
            shots = few_shot_pool if variant.startswith("fs_") else []
            preds, raws = run_one_cell(
                model, tokenizer, model_cfg, variant, test, shots, cfg.generation
            )
            m = compute_metrics(y_true, preds)
            print(f"[{model_cfg.short}/{variant}] acc={m['accuracy']:.3f} "
                  f"f1={m['macro_f1']:.3f}  unknown={m['unknown_rate']:.2%}")

            # Save predictions
            pred_path = out_dir / "predictions" / f"{model_cfg.short}_{variant}.jsonl"
            with pred_path.open("w", encoding="utf-8") as f:
                for ex, raw, pred in zip(test, raws, preds):
                    f.write(json.dumps({
                        "idx": ex.idx, "text": ex.text, "gold": ex.label,
                        "raw": raw[:500], "pred": pred,
                    }, ensure_ascii=False) + "\n")

            all_preds[f"{model_cfg.short}|{variant}"] = preds
            all_metrics.append({
                "model": model_cfg.short, "prompt": variant,
                "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                "f1_negative": m["f1_negative"], "f1_neutral": m["f1_neutral"],
                "f1_positive": m["f1_positive"],
                "unknown_rate": m["unknown_rate"], "n": m["n"],
            })
            all_errors[f"{model_cfg.short}|{variant}"] = analyze(test, preds)

        unload(model)

    # ---- Write metrics table
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    print(f"\nWrote {out_dir / 'metrics.csv'}")

    # ---- Save error analysis
    with (out_dir / "error_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(all_errors, f, ensure_ascii=False, indent=2)

    # ---- Paired significance: for each model, en vs ur within each setting
    sig_rows = []
    for model_cfg in cfg.models:
        if args.only and model_cfg.short != args.only:
            continue
        for pair in [("zs_en", "zs_ur"), ("fs_en", "fs_ur"), ("cot_en", "cot_ur")]:
            ka, kb = f"{model_cfg.short}|{pair[0]}", f"{model_cfg.short}|{pair[1]}"
            if ka in all_preds and kb in all_preds:
                t = mcnemar_test(y_true, all_preds[ka], all_preds[kb])
                sig_rows.append({
                    "model": model_cfg.short, "setting": pair[0].split("_")[0],
                    "a": pair[0], "b": pair[1], **t,
                })
    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(out_dir / "significance.csv", index=False)

    # ---- Save y_true + a registry of prediction keys (for downstream scripts)
    with (out_dir / "y_true.json").open("w", encoding="utf-8") as f:
        json.dump(y_true, f)
    with (out_dir / "all_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(all_preds, f, ensure_ascii=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
