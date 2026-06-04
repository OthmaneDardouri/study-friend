"""
Model loading and inference routing.

Author: Othmane Dardouri
"""

from __future__ import annotations

from .utils import get_parent_model_path, get_device, Engine
from .engine.transformers import load_model_transformers, query_transformers
from .engine.mlx_vlm      import load_model_mlx,          query_mlx


def load_model(engine: Engine, model_path: str, verbose: bool = False):
    """Load the appropriate VLM based on the selected *engine*."""
    device = get_device()
    if verbose:
        print(f"Loading '{model_path}' via {engine} on {device}")

    match engine:
        case Engine.TRANSFORMERS:
            parent_path = get_parent_model_path(model_path)
            if verbose:
                print(f"  Parent path: {parent_path}")
            return load_model_transformers(model_path, parent_path, device, verbose)
        case Engine.MLX_VLM:
            return load_model_mlx(model_path, verbose)
        case _:
            raise ValueError(f"Unsupported engine: {engine!r}")


def query(
    engine: Engine,
    model,
    processor,
    config,
    prompt: str,
    images: list[str],
    temperature: float = 0.1,
    max_tokens: int = 999,
    verbose: bool = False,
) -> str:
    """Run inference on *images* using *prompt* and return the generated text."""
    device = get_device()

    match engine:
        case Engine.TRANSFORMERS:
            return query_transformers(
                model, processor, config, prompt, images,
                temperature, max_tokens, device, verbose,
            )
        case Engine.MLX_VLM:
            return query_mlx(
                model, processor, config, prompt, images,
                temperature, max_tokens, verbose,
            )
        case _:
            raise ValueError(f"Unsupported engine: {engine!r}")
