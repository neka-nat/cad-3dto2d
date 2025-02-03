import os
from typing import Literal

from build123d import (
    import_step,
    import_svg,
    Axis,
    Compound,
    ExportDXF,
    ExportSVG,
    LineType,
    Unit,
    ShapeList,
    Wire,
    Face,
)
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg

ImageTypes = Literal["svg", "png", "jpg", "jpeg"]


def _load_template() -> ShapeList[Wire | Face]:
    current_dir = os.path.dirname(__file__)
    template_file = os.path.join(current_dir, "templates/A4_LandscapeTD.svg")
    return import_svg(template_file)


def convert_2d_drawing(
    step_file: str,
    output_file: str,
    line_weight: float = 0.5,
    add_template: bool = True,
    x_offset: float = 0,
    y_offset: float = 0,
) -> None:
    model = import_step(step_file)
    view_port_org_front = (0, 0, 1000000)
    view_port_org_x_side = (1000000, 0, 0)
    view_port_org_y_side = (0, 1000000, 0)
    filename, ext = os.path.splitext(output_file)
    if ext == ".dxf":
        exporter = ExportDXF(unit=Unit.MM, line_weight=line_weight)
    elif ext[1:] in ImageTypes.__args__:
        exporter = ExportSVG(unit=Unit.MM, line_weight=line_weight)
    else:
        raise ValueError(f"Invalid export file type: {ext}")
    exporter.add_layer("visible", line_type=LineType.CONTINUOUS)
    exporter.add_layer("hidden", line_type=LineType.DASHED)
    if add_template:
        exporter.add_layer("template", line_type=LineType.CONTINUOUS)
        template = _load_template()
        tmp_size = Compound(children=template).bounding_box().size
        template = [shape.translate((-tmp_size.X / 2 + x_offset, -tmp_size.Y / 2 + y_offset, 0)) for shape in template]
        exporter.add_shape(template, layer="template")

    visibles = []
    hiddens = []

    front_visible, front_hidden = model.project_to_viewport(view_port_org_front)
    visibles.extend(front_visible)
    hiddens.extend(front_hidden)
    front_size = Compound(children=front_visible + front_hidden).bounding_box().size
    front_size_with_margin = (front_size.X * 1.1, front_size.Y * 1.1)
    side_x_visible, side_x_hidden = model.project_to_viewport(view_port_org_x_side)
    side_x_visible = [
        shape.rotate(Axis.Z, 90).translate((front_size_with_margin[0], 0, 0)) for shape in side_x_visible
    ]
    side_x_hidden = [shape.rotate(Axis.Z, 90).translate((front_size_with_margin[0], 0, 0)) for shape in side_x_hidden]
    visibles.extend(side_x_visible)
    hiddens.extend(side_x_hidden)
    side_y_visible, side_y_hidden = model.project_to_viewport(view_port_org_y_side)
    side_y_visible = [shape.translate((0, -front_size_with_margin[1], 0)) for shape in side_y_visible]
    side_y_hidden = [shape.translate((0, -front_size_with_margin[1], 0)) for shape in side_y_hidden]
    visibles.extend(side_y_visible)
    hiddens.extend(side_y_hidden)

    three_view_size = Compound(children=visibles + hiddens).bounding_box().size
    visibles = [
        shape.translate((-(three_view_size.X - front_size.X) / 2, (three_view_size.Y - front_size.Y) / 2, 0))
        for shape in visibles
    ]
    hiddens = [
        shape.translate((-(three_view_size.X - front_size.X) / 2, (three_view_size.Y - front_size.Y) / 2, 0))
        for shape in hiddens
    ]

    exporter.add_shape(visibles, layer="visible")
    exporter.add_shape(hiddens, layer="hidden")

    if ext[1:] in ImageTypes.__args__:
        svg_file = filename + ".svg"
        exporter.write(svg_file)
        drawing = svg2rlg(svg_file)
        renderPM.drawToFile(drawing, output_file, fmt=ext[1:])
    else:
        exporter.write(output_file)
