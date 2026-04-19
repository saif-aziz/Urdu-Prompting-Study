"""LLM loading and batched inference.

Key details:
- Uses HF chat templates when available (correct special tokens for Qwen/Gemma).
- 4-bit quantization via bitsandbytes for Gemma-2B to fit on T4.
- Left-padding + batched generate for throughput.
- Greedy decoding (do_sample=False) for reproducibility.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import gc
import torch
from tqdm.auto import tqdm


def load_model_and_tokenizer(model_cfg, dtype=None):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # crucial for batched generate

    quant_config = None
    if model_cfg.quantize:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        trust_remote_code=True,
        torch_dtype=dtype or (torch.float16 if torch.cuda.is_available() else torch.float32),
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quant_config,
    )
    model.eval()
    return model, tokenizer


def _format_chat(tokenizer, system_msg: str, user_msg: str, use_chat_template: bool) -> str:
    """Build the final input string. Some tokenizers (Gemma) do not allow a separate
    'system' role, so we fold system into the user turn for those."""
    if use_chat_template and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        # Gemma chat template disallows system role — merge if needed
        name = getattr(tokenizer, "name_or_path", "").lower()
        if "gemma" in name:
            messages = [{"role": "user", "content": f"{system_msg}\n\n{user_msg}"}]
        else:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Raw fallback
    return f"{system_msg}\n\n{user_msg}\n"


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[Dict[str, str]],
    *,
    use_chat_template: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    batch_size: int,
) -> List[str]:
    """Generate completions for a list of {'system', 'user'} prompts."""
    formatted = [_format_chat(tokenizer, p["system"], p["user"], use_chat_template) for p in prompts]
    outputs: List[str] = []
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(formatted), batch_size), desc="generate", leave=False):
        batch = formatted[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # Slice off input ids
        for j in range(gen.shape[0]):
            input_len = enc["input_ids"][j].shape[0]
            generated_ids = gen[j][input_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            outputs.append(text)
    return outputs


def unload(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
