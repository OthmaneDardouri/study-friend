"""
Flask-based Markdown viewer with inline image support.

Author: Othmane Dardouri
"""

from __future__ import annotations

import argparse
import os
import re
import urllib.parse
import webbrowser
from pathlib import Path

import markdown
from flask import Flask

from .utils import add_argument_common, add_argument_display, print_args, extract_url
from .constants import DEFAULT_HTML_DIV, DEFAULT_HTML_IMAGE, DEFAULT_HTML_STYLE

_page_cache: dict[str, str] = {}


def _image_tag(src: str) -> str:
    return DEFAULT_HTML_IMAGE.format(src=urllib.parse.quote(src))


def _wrap_images(paths: list[str]) -> str:
    return DEFAULT_HTML_DIV.format(images="".join(_image_tag(p) for p in paths))


def make_routes(app: Flask, html: str) -> None:
    """Register Flask routes that serve pre-rendered HTML pages."""
    _page_cache["index"] = DEFAULT_HTML_STYLE + html

    @app.route("/")
    def index():
        return _page_cache.get("index", "<h1>No content loaded.</h1>")

    @app.route("/<path:page>")
    def page(page: str):
        return _page_cache.get(page, "<h1>Page not found.</h1>")


def render_markdown_to_html(md_text: str, image_base_dir: str = "") -> str:
    """Convert Markdown (with image references) to a full HTML string."""
    def replace_img(match):
        src       = match.group(1)
        full_path = os.path.join(image_base_dir, src) if image_base_dir else src
        return _image_tag(full_path)

    md_text = re.sub(r"!\[.*?\]\((.*?)\)", replace_img, md_text)
    body    = markdown.markdown(md_text, extensions=["extra", "codehilite"])
    return DEFAULT_HTML_STYLE + body


def display(file_path: str, url: str = "http://127.0.0.1:5000", here: bool = False) -> None:
    """Serve *file_path* as rendered HTML, or render inline in Jupyter."""
    text = Path(file_path).read_text(encoding="utf-8")
    html = render_markdown_to_html(text, image_base_dir=str(Path(file_path).parent))

    if here:
        try:
            from IPython.display import HTML, display as ipy_display
            ipy_display(HTML(html))
        except ImportError:
            print(html)
        return

    app  = Flask(__name__)
    make_routes(app, html)
    host, port = extract_url(url)
    webbrowser.open(url)
    app.run(host=host, port=int(port) if port else 5000, debug=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display a Markdown file in the browser.")
    add_argument_common(parser)
    add_argument_display(parser)
    args = parser.parse_args()
    if args.verbose:
        print_args(args)
    display(args.file, args.url, args.here)


if __name__ == "__main__":
    main()
