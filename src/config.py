"""Config loading. Keep everything typed so IDEs/Colab behave."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class DatasetCfg:
    hf_candidates: List[str]
    local_csv: str
    text_column: str
    label_column: str
    max_train: int
    max_test: int
    label_set: List[str]


@dataclass
class ModelCfg:
    name: str
    short: str
    quantize: bool
    chat_template: bool


@dataclass
class BaselineXLMRCfg:
    name: str
    epochs: int
    batch_size: int
    lr: float
    max_length: int


@dataclass
class FewShotCfg:
    k: int
    balanced: bool


@dataclass
class GenerationCfg:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    batch_size: int


@dataclass
class AblationsCfg:
    shot_counts: List[int]


@dataclass
class Config:
    seed: int
    output_dir: str
    dataset: DatasetCfg
    models: List[ModelCfg]
    baseline_xlmr: BaselineXLMRCfg
    prompts: List[str]
    few_shot: FewShotCfg
    generation: GenerationCfg
    ablations: AblationsCfg


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        seed=raw["seed"],
        output_dir=raw["output_dir"],
        dataset=DatasetCfg(**raw["dataset"]),
        models=[ModelCfg(**m) for m in raw["models"]],
        baseline_xlmr=BaselineXLMRCfg(**raw["baseline_xlmr"]),
        prompts=raw["prompts"],
        few_shot=FewShotCfg(**raw["few_shot"]),
        generation=GenerationCfg(**raw["generation"]),
        ablations=AblationsCfg(**raw["ablations"]),
    )
