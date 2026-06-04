"""
PDF-to-image conversion utilities.

Author: Othmane Dardouri
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pdf2image
import PIL.Image
from tqdm import tqdm

from .utils import add_argument_common, add_argument_convert


def save_images(
    output_dir: str | Path,
    images: list[PIL.Image.Image],
    names: list[str],
    max_size: int,
    verbose: bool = False,
) -> None:
    """Resize (if needed) and save *images* as JPEG files inside *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Saving {len(images)} image(s) to '{out}'")

    for name, img in zip(names, images):
        scale = max(1.0, max(img.size) / max_size)
        if scale > 1.0:
            img = img.resize(
                (int(img.width / scale), int(img.height / scale)),
                PIL.Image.LANCZOS,
            )
        img.save(out / f"{name}.jpeg")


def convert_pdfs_to_images(
    directory: str | Path,
    max_size: int,
    verbose: bool = False,
) -> list[str]:
    """Convert every PDF in *directory* to JPEG images.

    Returns a list of output sub-directory paths (one per PDF).
    """
    directory   = Path(directory)
    output_dirs: list[str] = []

    for pdf_path in tqdm(sorted(directory.iterdir()), desc="PDF -> images"):
        if pdf_path.suffix.lower() != ".pdf" or not pdf_path.is_file():
            continue
        if verbose:
            print(f"Converting {pdf_path.name}")

        sub_dir = pdf_path.with_suffix("")
        images  = pdf2image.convert_from_path(str(pdf_path))
        names   = [str(i) for i in range(len(images))]
        save_images(sub_dir, images, names, max_size, verbose)
        output_dirs.append(str(sub_dir))

    return output_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDFs to images.")
    add_argument_common(parser)
    add_argument_convert(parser)
    args = parser.parse_args()
    convert_pdfs_to_images(args.dir, args.image_size, args.verbose)


if __name__ == "__main__":
    main()
