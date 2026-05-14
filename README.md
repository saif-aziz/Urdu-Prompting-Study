# Urdu Prompting Study

**Research question:** Does prompting in Urdu vs. English yield better performance for Urdu-native tasks on small multilingual LMs?

**Task:** Urdu sentiment classification (positive / negative / neutral).

**Models compared:**
- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `xlm-roberta-base` (fine-tuned supervised baseline)
- Majority-class baseline

**Prompt variants (the core ablation):**
1. `zs_en` — zero-shot, English instruction
2. `zs_ur` — zero-shot, Urdu instruction
3. `fs_en` — 3-shot, English instruction + English reasoning
4. `fs_ur` — 3-shot, Urdu instruction + Urdu reasoning
5. `cot_en` — English CoT ("think step by step")
6. `cot_ur` — Urdu CoT ("قدم بہ قدم سوچیں")

6 prompts × 3 LMs + 2 baselines = **20 experimental cells**.

## Quick start

### Colab (recommended — T4 GPU is enough)
Open `notebooks/colab_runner.ipynb` in Colab, set runtime to T4, run all cells.

### Local
```bash
pip install -r requirements.txt
python -m src.main --config configs/default.yaml
```

End-to-end run on full test set: ~3–4 hours on T4. For a smoke test:
```bash
python -m src.main --config configs/default.yaml --max-test 50
```

## What gets produced

After a full run, `results/` contains:
- `predictions/{model}_{prompt}.jsonl` — raw predictions
- `metrics.csv` — accuracy / macro-F1 per cell
- `error_analysis.json` — bucketed error counts per cell
- `confusion_matrices.png` — 20 small confusion plots
- `main_results_table.tex` — drop straight into the paper
- `heatmap_en_vs_ur.png` — the money figure for the paper

## Project structure

```
src/
  config.py           # dataclasses + YAML loader
  data.py             # dataset loading + few-shot selection
  prompts.py          # all 6 prompt templates
  inference.py        # generate + parse labels
  baseline_finetune.py # XLM-R fine-tuning baseline
  metrics.py          # accuracy, F1, McNemar
  error_analysis.py   # auto-buckets errors (negation, code-mix, script, etc.)
  main.py             # orchestrates everything
scripts/
  make_figures.py     # heatmap + confusion matrices + LaTeX table
paper/
  main.tex            # IEEE template, wired to auto-generated figures
  references.bib
```

