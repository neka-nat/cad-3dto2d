import os
from typing import Iterable

from build123d import Compound, Edge, ShapeList, Wire, Face, import_step, import_svg

from .exporters.dxf import export_dxf_layers
from .exporters.svg import export_svg_layers, inject_svg_text, rasterize_svg
from .annotations.dimensions import (
    DimensionSettings,
    DimensionText,
    default_settings_from_size,
    generate_basic_dimensions,
)
from .layout import layout_three_views
from .styles import load_style
from .templates import TemplateSpec, load_template
from .views import project_three_views

Shape = Wire | Face | Edge
RASTER_IMAGE_TYPES = {"png", "jpg", "jpeg"}


def _load_template(template_spec: TemplateSpec) -> ShapeList[Wire | Face]:
    return import_svg(template_spec.file_path)


def _normalize_outputs(output_file: str, output_files: Iterable[str] | None) -> list[str]:
    outputs = [output_file]
    if output_files:
        outputs.extend(output_files)
    seen: set[str] = set()
    unique: list[str] = []
    for path in outputs:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _centered_frame_bounds(template_spec: TemplateSpec | None) -> tuple[float, float, float, float] | None:
    if not template_spec or not template_spec.frame_bbox_mm:
        return None
    min_x, min_y, max_x, max_y = template_spec.frame_bbox_mm
    if template_spec.paper_size_mm:
        paper_w, paper_h = template_spec.paper_size_mm
        return (
            min_x - paper_w / 2,
            min_y - paper_h / 2,
            max_x - paper_w / 2,
            max_y - paper_h / 2,
        )
    return (min_x, min_y, max_x, max_y)


def _clamp_offset(
    base: float,
    direction: int,
    frame_min: float,
    frame_max: float,
    offset: float,
    padding: float = 0.0,
) -> float:
    if direction >= 0:
        available = frame_max - base - padding
        return min(offset, max(0.0, available))
    available = base - frame_min - padding
    return -min(offset, max(0.0, available))


def _build_layers(
    model,
    add_template: bool,
    template_spec: TemplateSpec | None,
    x_offset: float,
    y_offset: float,
    add_dimensions: bool,
    dimension_settings: DimensionSettings | None,
) -> tuple[dict[str, list[Shape]], list[DimensionText]]:
    layers: dict[str, list[Shape]] = {}
    dim_texts: list[DimensionText] = []
    views = project_three_views(model)
    layout = layout_three_views(
        views.front,
        views.side_x,
        views.side_y,
        frame_bbox_mm=template_spec.frame_bbox_mm if template_spec else None,
        paper_size_mm=template_spec.paper_size_mm if template_spec else None,
        scale=template_spec.default_scale if template_spec else None,
    )
    layers["visible"] = layout.combined.visible
    layers["hidden"] = layout.combined.hidden
    if add_dimensions:
        frame_bounds = _centered_frame_bounds(template_spec)
        dims: list[Shape] = []
        for view, horizontal_dir, vertical_dir in (
            (layout.front, 1, 1),
            (layout.side_x, 1, 1),
            (layout.side_y, -1, 1),
        ):
            shapes = view.visible + view.hidden
            if not shapes:
                continue
            bounds = Compound(children=shapes).bounding_box()
            size = bounds.size
            settings = dimension_settings or default_settings_from_size(size.X, size.Y)
            horizontal_offset = None
            vertical_offset = None
            if frame_bounds:
                frame_min_x, frame_min_y, frame_max_x, frame_max_y = frame_bounds
                base_y = bounds.max.Y if horizontal_dir >= 0 else bounds.min.Y
                base_x = bounds.max.X if vertical_dir >= 0 else bounds.min.X
                padding = settings.text_gap + settings.text_height
                horizontal_offset = _clamp_offset(
                    base_y,
                    horizontal_dir,
                    frame_min_y,
                    frame_max_y,
                    settings.offset,
                    padding=padding,
                )
                vertical_offset = _clamp_offset(
                    base_x,
                    vertical_dir,
                    frame_min_x,
                    frame_max_x,
                    settings.offset,
                    padding=padding,
                )
            result = generate_basic_dimensions(
                shapes,
                settings=settings,
                horizontal_dir=horizontal_dir,
                vertical_dir=vertical_dir,
                horizontal_offset=horizontal_offset,
                vertical_offset=vertical_offset,
            )
            dims.extend(result.lines)
            dim_texts.extend(result.texts)
        if dims:
            layers["dims"] = dims
    if add_template and template_spec:
        template = _load_template(template_spec)
        tmp_size = Compound(children=template).bounding_box().size
        template = [
            shape.translate((-tmp_size.X / 2 + x_offset, -tmp_size.Y / 2 + y_offset, 0)) for shape in template
        ]
        layers["template"] = template
    return layers, dim_texts


def _export_layers(
    layers: dict[str, list[Shape]],
    output_file: str,
    line_weight: float,
    line_types: dict[str, "LineType"] | None,
    text_annotations: list[DimensionText] | None,
) -> None:
    _, ext = os.path.splitext(output_file)
    ext = ext.lower()
    if ext == ".dxf":
        export_dxf_layers(layers, output_file, line_weight, line_types=line_types)
        return
    if ext == ".svg":
        export_svg_layers(layers, output_file, line_weight, line_types=line_types)
        if text_annotations:
            inject_svg_text(output_file, text_annotations)
        return
    if ext[1:] in RASTER_IMAGE_TYPES:
        svg_file = os.path.splitext(output_file)[0] + ".svg"
        export_svg_layers(layers, svg_file, line_weight, line_types=line_types)
        if text_annotations:
            inject_svg_text(svg_file, text_annotations)
        rasterize_svg(svg_file, output_file)
        return
    raise ValueError(f"Invalid export file type: {ext}")


def convert_2d_drawing(
    step_file: str,
    output_file: str,
    line_weight: float = 0.5,
    add_template: bool = True,
    template_name: str = "A4_LandscapeTD",
    x_offset: float = 0,
    y_offset: float = 0,
    style_name: str | None = "iso",
    add_dimensions: bool = False,
    dimension_settings: DimensionSettings | None = None,
    output_files: Iterable[str] | None = None,
) -> None:
    model = import_step(step_file)
    line_types = load_style(style_name).resolve_line_types() if style_name else None
    template_spec = load_template(template_name)
    layers, text_annotations = _build_layers(
        model,
        add_template,
        template_spec,
        x_offset,
        y_offset,
        add_dimensions,
        dimension_settings,
    )
    for target in _normalize_outputs(output_file, output_files):
        _export_layers(layers, target, line_weight, line_types=line_types, text_annotations=text_annotations)
