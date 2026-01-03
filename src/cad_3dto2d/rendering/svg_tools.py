from __future__ import annotations

import os

from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg


def rasterize_svg(svg_file: str, output_file: str) -> None:
    drawing = svg2rlg(svg_file)
    _, ext = os.path.splitext(output_file)
    renderPM.drawToFile(drawing, output_file, fmt=ext.lstrip(".").lower())
