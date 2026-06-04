"""
Constants and configuration for StudyFriend.

Author: Othmane Dardouri
"""

from __future__ import annotations

import torch
from enum import StrEnum
from typing import Final


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Engine(StrEnum):
    """Available inference backends."""
    TRANSFORMERS = "transformers"
    MLX_VLM      = "mlx_vlm"


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

DEVICE_IS_MLX: Final[bool] = torch.backends.mps.is_available()
DEVICE_MEMORY_GB: Final[float] = (
    torch.mps.recommended_max_memory() if DEVICE_IS_MLX
    else (torch.cuda.mem_get_info()[1] if torch.cuda.is_available() else 0)
) / (1 << 30)


# ---------------------------------------------------------------------------
# Image / conversion defaults
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_SIZE: Final[int] = 500          # px — good for 3B/7B VLMs


# ---------------------------------------------------------------------------
# Query defaults
# ---------------------------------------------------------------------------

DEFAULT_GROUP_SIZE: Final[int]       = 3 if DEVICE_MEMORY_GB > 4.0 else 1
DEFAULT_TEMPERATURE: Final[float]    = 0.1
DEFAULT_MAX_TOKENS: Final[int]       = 999
DEFAULT_COUNTER_INJECTOR: Final[str] = "<N>"

DEFAULT_MODEL: Final[str] = (
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    if DEVICE_IS_MLX else (
        "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"
        if DEVICE_MEMORY_GB > 4.0
        else "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit"
    )
)
DEFAULT_ENGINE: Final[Engine] = Engine.MLX_VLM if DEVICE_IS_MLX else Engine.TRANSFORMERS

DEFAULT_PLURALITY_INJECTORS: Final[list[str]] = [
    f"<{DEFAULT_COUNTER_INJECTOR}>",
    f"for each of the {DEFAULT_COUNTER_INJECTOR} slides",
]
DEFAULT_SINGULARITY_INJECTORS: Final[list[str]] = ["1", ""]

DEFAULT_TITLE_PROMPT: Final[str] = (
    "\nWhat is the slide title and subtitle (leave subtitle blank if absent)? "
    "Answer using exactly this template and nothing else:\n <Title> - <Subtitle>"
)
DEFAULT_QUESTION_PROMPT: Final[str] = (
    f"\nExample:\n### Slide {DEFAULT_PLURALITY_INJECTORS[0]}:\n\n1. ?\n2. ?\n\n"
    "What is the subject of the slides? Generate 2 distinct questions "
    f"{DEFAULT_PLURALITY_INJECTORS[1]} about charts and concepts. "
    "Do not provide answers. Use the template above."
)
DEFAULT_ANSWER_PROMPT: Final[str] = (
    "\nLook at the images and briefly answer:\n{question}"
)
DEFAULT_MATH_REGEX: Final[str] = r"( *\\\[[^\S\r\n]*[\s\S]+?\\\]| *\\\([^\S\r\n]*[\s\S]+?\\\))"
OUTPUT_FILE: Final[str] = "output.md"

DEFAULT_MODEL_FINETUNING_NAMES: Final[list[str]] = ["unsloth", "bnb", "4bit", "awq", "gptq"]
DEFAULT_MODEL_REGEX: Final[str] = r"[\w.-]+/[\w.-]+(?<![{model_finetunigs}])"


# ---------------------------------------------------------------------------
# Display defaults
# ---------------------------------------------------------------------------

DEFAULT_URL: Final[str]       = "http://127.0.0.1:5000"
DEFAULT_URL_REGEX: Final[str] = r"^(https?://)?([a-zA-Z0-9.-])+(:[0-9]+)?$"

DEFAULT_HTML_IMAGE: Final[str] = '<img src="{src}" style="max-width:100%;">'
DEFAULT_HTML_DIV: Final[str]   = '<div class="slide-group">{images}</div>'
DEFAULT_HTML_STYLE: Final[str] = """
<style>
  body {{ font-family: sans-serif; max-width: 960px; margin: auto; padding: 1rem; }}
  .slide-group {{ border: 1px solid #ddd; border-radius: 6px; padding: 1rem; margin: 1rem 0; }}
  footer {{ text-align: center; color: #888; font-size: 0.8rem; margin-top: 2rem; }}
</style>
<footer>Made with StudyFriend 📚 — by Othmane Dardouri</footer>
"""
