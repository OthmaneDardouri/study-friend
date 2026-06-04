"""
Apple MLX inference backend.

Author: Othmane Dardouri
"""

from __future__ import annotations

import torch
from ..utils import get_device

_IS_MLX = get_device() == torch.device("mps")

if _IS_MLX:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    def load_model_mlx(model_path: str, verbose: bool = False):
        """Load an MLX-compatible VLM."""
        if verbose:
            print(f"Loading MLX model from {model_path!r}")
        model, processor = load(model_path)
        config = load_config(model_path)
        return model, processor, config

    def query_mlx(
        model,
        processor,
        config,
        prompt: str,
        images: list,
        temperature: float = 0.1,
        max_tokens: int = 999,
        verbose: bool = False,
    ) -> str:
        """Run inference using the MLX backend."""
        formatted = apply_chat_template(processor, config, prompt, num_images=len(images))
        return generate(
            model, processor, formatted, images,
            temperature=temperature, max_tokens=max_tokens, verbose=verbose,
        )

else:
    def load_model_mlx(*_, **__):
        raise NotImplementedError("MLX backend requires Apple Silicon with MPS support.")

    def query_mlx(*_, **__):
        raise NotImplementedError("MLX backend requires Apple Silicon with MPS support.")
