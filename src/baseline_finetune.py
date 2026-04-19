"""Supervised baseline: fine-tune XLM-RoBERTa on the same train split.

This gives us the 'what prompting has to beat' number. Also exposes the claim
'small LMs with good Urdu prompts approach fine-tuned XLM-R' — or don't.
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import torch

from .data import Example, LABELS, LABEL2ID, ID2LABEL


def train_and_eval_xlmr(
    cfg,
    train: List[Example],
    test: List[Example],
) -> Dict:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )
    from datasets import Dataset

    tok = AutoTokenizer.from_pretrained(cfg.baseline_xlmr.name)

    def to_ds(exs: List[Example]) -> Dataset:
        return Dataset.from_dict({
            "text": [e.text for e in exs],
            "label": [LABEL2ID[e.label] for e in exs],
        })

    train_ds = to_ds(train)
    test_ds = to_ds(test)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=cfg.baseline_xlmr.max_length)

    train_ds = train_ds.map(tok_fn, batched=True)
    test_ds = test_ds.map(tok_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.baseline_xlmr.name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir="results/_xlmr_tmp",
        per_device_train_batch_size=cfg.baseline_xlmr.batch_size,
        per_device_eval_batch_size=cfg.baseline_xlmr.batch_size,
        num_train_epochs=cfg.baseline_xlmr.epochs,
        learning_rate=cfg.baseline_xlmr.lr,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
    )
    trainer.train()

    preds_out = trainer.predict(test_ds)
    pred_ids = np.argmax(preds_out.predictions, axis=-1)
    preds = [ID2LABEL[int(i)] for i in pred_ids]
    return {"predictions": preds}
