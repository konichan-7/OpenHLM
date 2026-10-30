#!/usr/bin/env python3
"""Fetch and vector-crop four cited figures from the pinned arXiv paper.

Requires pypdf and Poppler's pdftocairo/pdfinfo. Generated assets are ignored by
Git but should be included in a self-contained LaTeX delivery archive. No paper
TeX is executed. Original figure content is not replotted or rasterized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request

from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject


def checked_pdf(path: Path, expected_sha256: str) -> PdfReader:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Paper checksum mismatch: expected {expected_sha256}, got {digest}")
    reader = PdfReader(path)
    if len(reader.pages) != 19:
        raise ValueError(f"Expected 19 paper pages, got {len(reader.pages)}")
    return reader


def crop_figure(reader: PdfReader, figure: dict, destination: Path, temporary: Path) -> None:
    page = reader.pages[figure["page"] - 1]
    left, top, right, bottom = figure["bbox_top_left_pt"]
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"Invalid crop box for figure {figure['number']}")
    writer = PdfWriter()
    output_page = writer.add_page(page)
    box = RectangleObject([left, height - bottom, right, height - top])
    output_page.mediabox = box
    output_page.cropbox = box
    if "/Annots" in output_page:
        del output_page["/Annots"]
    cropped = temporary / figure["file"]
    with cropped.open("wb") as stream:
        writer.write(stream)
    # Re-emit only visible drawing operations. Merely changing CropBox leaves
    # hidden paragraphs in extraction/search and can pollute the final report.
    subprocess.run(
        ["pdftocairo", "-pdf", "-origpagesizes", str(cropped), str(destination)],
        check=True, capture_output=True, text=True,
    )
    extracted = PdfReader(destination)
    if len(extracted.pages) != 1:
        raise ValueError(f"Figure {figure['number']} is not a single-page asset")
    actual = extracted.pages[0].mediabox
    if abs(float(actual.width) - (right-left)) > .1 or abs(float(actual.height) - (bottom-top)) > .1:
        raise ValueError(f"Unexpected dimensions for figure {figure['number']}")
    text = extracted.pages[0].extract_text()
    for required in figure["visible_text_checks"]:
        if required not in text:
            raise ValueError(f"Missing figure label {required!r} in figure {figure['number']}")
    for forbidden in ("Figure 2:", "Figure 3:", "Figure 4:", "Figure 5:", "First-order gradients"):
        if forbidden in text:
            raise ValueError(f"Unwanted caption/body text retained in figure {figure['number']}")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, help="Use an already-downloaded, checksum-matching PDF")
    parser.add_argument("--output-dir", type=Path, default=root / "figures")
    args = parser.parse_args()
    for executable in ("pdftocairo",):
        if shutil.which(executable) is None:
            raise SystemExit(f"Missing dependency: {executable}")
    manifest = json.loads((root / "figure_sources.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="supermimic-figures-") as tmp:
        temporary = Path(tmp)
        paper = args.paper or temporary / "paper.pdf"
        if args.paper is None:
            request = urllib.request.Request(manifest["source_pdf_url"], headers={"User-Agent": "SuperMimicReport/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                paper.write_bytes(response.read())
        reader = checked_pdf(paper, manifest["source_pdf_sha256"])
        for figure in manifest["figures"]:
            destination = args.output_dir / figure["file"]
            crop_figure(reader, figure, destination, temporary)
            print(f"Figure {figure['number']}: {destination} (vector PDF; labels and dimensions verified)")


if __name__ == "__main__":
    main()
