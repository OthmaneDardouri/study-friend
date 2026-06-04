"""
Utility helpers for StudyFriend.

Author: Othmane Dardouri
"""

from __future__ import annotations

import re
import torch
from .constants import (
    DEFAULT_URL_REGEX,
    DEFAULT_MATH_REGEX,
    DEFAULT_MODEL_FINETUNING_NAMES,
    DEFAULT_MODEL_REGEX,
    Engine,
)


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

def add_argument_common(parser) -> None:
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Enable verbose output.")


def add_argument_convert(parser) -> None:
    from .constants import DEFAULT_IMAGE_SIZE
    parser.add_argument("-im", "--image_size", type=int, default=DEFAULT_IMAGE_SIZE,
                        help="Maximum image dimension (px) after resize.")
    parser.add_argument("-d", "--dir", type=str, default=".",
                        help="Directory containing PDF files.")


def add_argument_query(parser) -> None:
    from .constants import (
        OUTPUT_FILE, DEFAULT_MODEL, DEFAULT_ENGINE,
        DEFAULT_TITLE_PROMPT, DEFAULT_QUESTION_PROMPT, DEFAULT_ANSWER_PROMPT,
        DEFAULT_GROUP_SIZE, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS,
        DEFAULT_COUNTER_INJECTOR, DEFAULT_SINGULARITY_INJECTORS,
        DEFAULT_PLURALITY_INJECTORS,
    )
    parser.add_argument("-o",  "--output_file",       type=str,   default=OUTPUT_FILE)
    parser.add_argument("-m",  "--model",              type=str,   default=DEFAULT_MODEL)
    parser.add_argument("-e",  "--engine",             type=str,   default=DEFAULT_ENGINE,
                        choices=[e.value for e in Engine])
    parser.add_argument("-id", "--image_dir",          type=str,   default="")
    parser.add_argument("-tp", "--title_prompt",       type=str,   default=DEFAULT_TITLE_PROMPT)
    parser.add_argument("-qp", "--question_prompt",    type=str,   default=DEFAULT_QUESTION_PROMPT)
    parser.add_argument("-aq", "--answer_prompt",      type=str,   default=DEFAULT_ANSWER_PROMPT)
    parser.add_argument("-g",  "--group_size",         type=int,   default=DEFAULT_GROUP_SIZE)
    parser.add_argument("-t",  "--temperature",        type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("-mt", "--max_tokens",         type=int,   default=DEFAULT_MAX_TOKENS)
    parser.add_argument("-ci", "--counter_injector",   type=str,   default=DEFAULT_COUNTER_INJECTOR)
    parser.add_argument("-si", "--singular_injectors", nargs="+",  default=DEFAULT_SINGULARITY_INJECTORS)
    parser.add_argument("-pi", "--plural_injectors",   nargs="+",  default=DEFAULT_PLURALITY_INJECTORS)


def add_argument_display(parser) -> None:
    from .constants import DEFAULT_URL
    parser.add_argument("-f",  "--file", type=str, required=True,
                        help="Markdown file to display.")
    parser.add_argument("-u",  "--url",  type=str, default=DEFAULT_URL)
    parser.add_argument("--here", action="store_true", default=False,
                        help="Render inline (useful inside Jupyter notebooks).")


def print_args(args) -> None:
    for key, val in vars(args).items():
        print(f"  {key}: {val}")


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

def extract_url(url: str) -> tuple[str, str | None]:
    """Return (host, port) from a URL string, or raise ValueError."""
    if not re.match(DEFAULT_URL_REGEX, url):
        raise ValueError(f"Invalid URL: {url!r}")
    url = url.split("://", 1)[-1]
    host, _, port = url.partition(":")
    return host, port or None


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def standardize_math_formulas(text: str) -> str:
    """Collapse multi-line LaTeX expressions to a single line."""
    for match in re.findall(DEFAULT_MATH_REGEX, text):
        text = text.replace(match, re.sub(r"\n", "", match).strip())
    return text


def prompt_injection(prompt: str, originals: list[str], replacements: list[str]) -> str:
    """Replace each *original* pattern in *prompt* with the corresponding *replacement*."""
    for orig, repl in zip(originals, replacements):
        prompt = re.sub(orig, repl, prompt)
    return prompt


# ---------------------------------------------------------------------------
# Device & model helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_string_variants(
    strings: list[str],
    sep: str,
    variants: list = [str.lower, str.capitalize, str.upper],
) -> str:
    """Return all case-variants of *strings* joined by *sep*.

    Example: (["monday"], "|") -> "monday|Monday|MONDAY"
    """
    return sep.join(sep.join(v(s) for s in strings) for v in variants)


def get_parent_model_path(model_path: str) -> str:
    """Attempt to strip fine-tune suffix and return the base model path."""
    finetuning_variants = get_string_variants(DEFAULT_MODEL_FINETUNING_NAMES, "|")
    pattern = DEFAULT_MODEL_REGEX.format(model_finetunigs=finetuning_variants)
    match = re.search(pattern, model_path)
    if match:
        return model_path[match.start():match.end()]
    print(f"Warning: could not extract parent model from {model_path!r}. Returning as-is.")
    return model_path
