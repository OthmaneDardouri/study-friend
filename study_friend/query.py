"""
Core query pipeline: PDF -> images -> VLM -> Markdown study notes.

Author: Othmane Dardouri
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
from natsort import natsorted
from tqdm import tqdm

from .utils import (
    add_argument_common,
    add_argument_convert,
    add_argument_query,
    prompt_injection,
    standardize_math_formulas,
    print_args,
)
from .models import load_model, query
from .convert import convert_pdfs_to_images


def group_images(
    image_dir: str | Path,
    group_size: int = 3,
    verbose: bool = False,
) -> list[list[str]]:
    """Partition sorted image files in *image_dir* into windows of *group_size*."""
    image_dir = Path(image_dir).resolve()
    if not image_dir.is_dir():
        return []
    if verbose:
        print(f"Grouping images in '{image_dir}' (group_size={group_size})")

    files      = natsorted(os.listdir(image_dir))
    n          = len(files)
    padded_len = n + (group_size - n % group_size) % group_size
    indices    = np.arange(padded_len).reshape(-1, group_size)

    groups: list[list[str]] = []
    for window in indices:
        group = [str(image_dir / files[i]) for i in window if i < n]
        if group:
            groups.append(group)
    return groups


def query_images(
    groups: list[list[str]],
    engine,
    model,
    processor,
    config,
    title_prompt: str,
    question_prompt: str,
    answer_prompt: str,
    counter: str,
    singularities: list[str],
    pluralities: list[str],
    output_file: str | Path,
    verbose: bool = False,
) -> None:
    """Query the VLM on each image group and append Q&A markdown to *output_file*."""
    with open(output_file, "a", encoding="utf-8") as fout:
        if groups and groups[0]:
            title = query(
                engine, model, processor, config,
                title_prompt, [groups[0][0]], verbose=verbose,
            )
            if verbose:
                print(f"Title: {title}")
            fout.write(f"# {title}\n")

        for images in tqdm(groups, leave=False, desc="Questions"):
            _prompt = question_prompt
            if verbose:
                print(f"Images: {images}")

            if len(images) == 1:
                _prompt = prompt_injection(_prompt, pluralities, singularities)
            _prompt = prompt_injection(_prompt, [counter], [str(len(images))])

            raw_output = query(
                engine, model, processor, config,
                _prompt, images, verbose=verbose,
            )
            fout.write(f"\nFiles: [{', '.join(images)}]\n")

            for line in tqdm(raw_output.split("\n"), leave=False, desc="Answers "):
                questions = re.findall(r"[\w\d].*\?$", line)
                if questions:
                    q      = questions[0]
                    answer = query(
                        engine, model, processor, config,
                        answer_prompt.format(question=q), images, verbose=verbose,
                    )
                    if verbose:
                        print(f"Q: {q}\nA: {answer}")
                    answer = standardize_math_formulas(answer)
                    fout.write(q + "\n" + answer + "\n")
                else:
                    fout.write(line + "\n")


def beautify_markdown(
    temp_file: str | Path,
    output_file: str | Path,
    verbose: bool = False,
) -> None:
    """Post-process raw VLM output into clean, readable Markdown."""
    temp_file   = Path(temp_file)
    output_file = Path(output_file)

    with temp_file.open("r", encoding="utf-8") as fin, \
         output_file.open("w", encoding="utf-8") as fout:

        for line in fin:
            questions = re.findall(r"[^\d. ].*\?$", line)
            slide     = re.search(r"(### Slide \d)(.*)$", line)
            files     = re.findall(r"Files: \[.*$", line)
            separator = re.findall(r"---$", line)

            if questions and not slide:
                fout.write(f"\n:question: {questions[0]}\n\n")
            elif slide:
                fout.write(f"### Slide {slide.group(2)}")
            elif files:
                fout.write(f"---\n\n{files[0]}\n")
            elif separator:
                pass
            else:
                fout.write(line)

    temp_file.unlink()
    if verbose:
        print(f"Output written to '{output_file}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Q&A study notes from PDF slides using a local VLM."
    )
    add_argument_common(parser)
    add_argument_convert(parser)
    add_argument_query(parser)
    args = parser.parse_args()

    if args.verbose:
        print("Options:")
        print_args(args)

    temp_file = args.output_file + "_temp"

    dirs = (
        convert_pdfs_to_images(args.dir, args.image_size, args.verbose)
        if not args.image_dir
        else [args.image_dir]
    )

    model, processor, config = load_model(args.engine, args.model, args.verbose)

    for doc_dir in tqdm(dirs, desc="Documents"):
        groups = group_images(doc_dir, args.group_size, args.verbose)
        query_images(
            groups, args.engine, model, processor, config,
            args.title_prompt, args.question_prompt, args.answer_prompt,
            args.counter_injector, args.singular_injectors, args.plural_injectors,
            temp_file, args.verbose,
        )

    beautify_markdown(temp_file, args.output_file, args.verbose)


if __name__ == "__main__":
    main()
