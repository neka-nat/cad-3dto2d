from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Iterable

from build123d import ExportSVG, LineType, Unit
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg

from ..types import Shape

if TYPE_CHECKING:
    from ..annotations.dimensions import DimensionText

_SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", _SVG_NS)

_LINE_TYPES: dict[str, LineType] = {
    "visible": LineType.CONTINUOUS,
    "hidden": LineType.DASHED,
    "template": LineType.CONTINUOUS,
}


def export_svg_layers(
    layers: dict[str, list[Shape]],
    output_file: str,
    line_weight: float,
    line_types: dict[str, LineType] | None = None,
) -> None:
    exporter = ExportSVG(unit=Unit.MM, line_weight=line_weight)
    resolved_types = dict(_LINE_TYPES)
    if line_types:
        resolved_types.update(line_types)
    layer_order = ["visible", "hidden", "template"]
    ordered_layers = [name for name in layer_order if name in layers]
    ordered_layers.extend(name for name in layers if name not in layer_order)
    for layer_name in ordered_layers:
        exporter.add_layer(layer_name, line_type=resolved_types.get(layer_name, LineType.CONTINUOUS))
    for layer_name in ordered_layers:
        exporter.add_shape(layers[layer_name], layer=layer_name)
    exporter.write(output_file)


def rasterize_svg(svg_file: str, output_file: str) -> None:
    drawing = svg2rlg(svg_file)
    _, ext = os.path.splitext(output_file)
    renderPM.drawToFile(drawing, output_file, fmt=ext.lstrip(".").lower())


def inject_svg_text(svg_file: str, texts: Iterable["DimensionText"], group_id: str = "dim_text") -> None:
    text_list = list(texts)
    if not text_list:
        return
    tree = ET.parse(svg_file)
    root = tree.getroot()
    transform_group = None
    for group in root.iter(f"{{{_SVG_NS}}}g"):
        transform = group.get("transform", "")
        if "scale(1,-1)" in transform:
            transform_group = group
            break
    if transform_group is None:
        transform_group = root

    for child in list(transform_group):
        if child.tag == f"{{{_SVG_NS}}}g" and child.get("id") == group_id:
            transform_group.remove(child)

    text_group = ET.SubElement(
        transform_group,
        f"{{{_SVG_NS}}}g",
        {"id": group_id, "fill": "rgb(0,0,0)"},
    )
    for text in text_list:
        attrs = {
            "x": f"{text.x}",
            "y": f"{-text.y}",
            "font-size": f"{text.height}",
            "font-family": "sans-serif",
            "text-anchor": text.anchor,
            "dominant-baseline": "central",
            "transform": "scale(1,-1)",
        }
        node = ET.SubElement(text_group, f"{{{_SVG_NS}}}text", attrs)
        node.text = text.text

    tree.write(svg_file, encoding="utf-8", xml_declaration=True)
