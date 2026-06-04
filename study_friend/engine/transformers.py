"""
HuggingFace Transformers inference backend.

Author: Othmane Dardouri
Inspired by: https://github.com/huggingface/transformers/pull/35569
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    GenerationConfig,
)


def _build_chat_messages(images: list, prompt: str) -> list[dict]:
    """Build a Qwen2.5-VL compatible chat message list."""
    return [{
        "role": "user",
        "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}],
    }]


def load_model_transformers(
    model_path: str,
    metadata_path: str,
    device: torch.device,
    verbose: bool = False,
):
    """Load a Qwen2.5-VL model and its processor/config."""
    if verbose:
        print(f"Loading Transformers model from {model_path!r}")
    model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map=device.type)
    processor = AutoProcessor.from_pretrained(metadata_path)
    config    = GenerationConfig.from_pretrained(metadata_path)
    return model, processor, config


def query_transformers(
    model,
    processor,
    config,
    prompt: str,
    image_paths: list[str],
    temperature: float = 0.1,
    max_tokens: int = 999,
    device: torch.device = torch.device("cuda"),
    verbose: bool = False,
) -> str:
    """Run one inference pass and return the decoded output string."""
    if not image_paths:
        return ""

    images      = [Image.open(p) for p in image_paths]
    text_prompt = processor.apply_chat_template(
        _build_chat_messages(images, prompt), add_generation_prompt=True
    )
    if verbose:
        print(f"Prompt: {text_prompt}")

    inputs = processor(
        text=[text_prompt], images=images, padding=True, return_tensors="pt"
    ).to(device)

    config.temperature = temperature
    output_ids = model.generate(**inputs, max_new_tokens=max_tokens, generation_config=config)
    new_token_ids = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, output_ids)
    ]
    return processor.batch_decode(
        new_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
