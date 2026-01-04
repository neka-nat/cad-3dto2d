from __future__ import annotations

from build123d import ExportDXF, LineType, Unit
import ezdxf
from ezdxf.addons.drawing import Frontend, RenderContext, layout
from ezdxf.addons.drawing.config import BackgroundPolicy, Configuration
from ezdxf.addons.drawing.svg import SVGBackend

from ..annotations.dimensions import DiameterDimensionSpec, DimensionSettings, LinearDimensionSpec
from ..templates import ParametricTemplateSpec, TemplateSpec
from ..types import BoundingBox2D, Shape

_LINE_TYPES: dict[str, LineType] = {
    "visible": LineType.CONTINUOUS,
    "hidden": LineType.DASHED,
    "template": LineType.CONTINUOUS,
    "title": LineType.CONTINUOUS,
}


def export_dxf_layers(
    layers: dict[str, list[Shape]],
    output_file: str,
    line_weight: float,
    line_types: dict[str, LineType] | None = None,
) -> None:
    exporter = ExportDXF(unit=Unit.MM, line_weight=line_weight)
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


def apply_template_to_doc(
    doc: "ezdxf.document.Drawing",
    template_spec: TemplateSpec,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> None:
    if isinstance(template_spec, ParametricTemplateSpec):
        _apply_parametric_template(doc, template_spec, x_offset=x_offset, y_offset=y_offset)


def _dimension_overrides(settings: DimensionSettings) -> dict[str, float | int | str]:
    override: dict[str, float | int | str] = {
        "dimtxt": settings.text_height,
        "dimasz": settings.arrow_size,
        "dimexo": settings.extension_offset if settings.extension_offset is not None else settings.extension_gap,
        "dimexe": settings.extension_extension if settings.extension_extension is not None else settings.extension_gap,
        "dimgap": settings.text_gap,
        "dimdec": settings.decimal_places,
    }
    if settings.arrow_block:
        override["dimblk"] = settings.arrow_block
    if settings.arrow_block1:
        override["dimblk1"] = settings.arrow_block1
    if settings.arrow_block2:
        override["dimblk2"] = settings.arrow_block2
    return override


def _apply_parametric_template(
    doc: "ezdxf.document.Drawing",
    template_spec: ParametricTemplateSpec,
    x_offset: float,
    y_offset: float,
) -> None:
    msp = doc.modelspace()
    _ensure_layer(doc, "template")
    _ensure_layer(doc, "title")
    frame_bbox = _centered_bbox(template_spec.frame_bbox_mm, template_spec.paper_size_mm)
    if frame_bbox:
        frame_bbox = _offset_bbox(frame_bbox, x_offset, y_offset)
        _add_bbox_rect(msp, frame_bbox, layer="template")
    if template_spec.title_block_bbox_mm:
        title_bbox = _centered_bbox(template_spec.title_block_bbox_mm, template_spec.paper_size_mm)
        if title_bbox:
            title_bbox = _offset_bbox(title_bbox, x_offset, y_offset)
            _add_bbox_rect(msp, title_bbox, layer="title")


def _centered_bbox(bbox: BoundingBox2D, paper_size_mm: tuple[float, float] | None) -> BoundingBox2D:
    min_x, min_y, max_x, max_y = bbox
    if not paper_size_mm:
        return bbox
    paper_w, paper_h = paper_size_mm
    return (
        min_x - paper_w / 2,
        min_y - paper_h / 2,
        max_x - paper_w / 2,
        max_y - paper_h / 2,
    )


def _offset_bbox(bbox: BoundingBox2D, x_offset: float, y_offset: float) -> BoundingBox2D:
    return (bbox[0] + x_offset, bbox[1] + y_offset, bbox[2] + x_offset, bbox[3] + y_offset)


def _add_bbox_rect(msp, bbox: BoundingBox2D, layer: str) -> None:
    min_x, min_y, max_x, max_y = bbox
    points = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]
    msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})


def _ensure_layer(doc: "ezdxf.document.Drawing", name: str) -> None:
    if not doc.layers.has_entry(name):
        doc.layers.new(name, dxfattribs={"linetype": "CONTINUOUS"})


def _set_dimension_text(dim_entity, text: str) -> None:
    if hasattr(dim_entity, "set_text"):
        dim_entity.set_text(text)
    else:
        dim_entity.dxf.text = text


def add_ezdxf_dimensions(
    doc: "ezdxf.document.Drawing",
    linear_dims: list[LinearDimensionSpec],
    diameter_dims: list[DiameterDimensionSpec],
    layer_name: str = "dims",
) -> None:
    if not doc.layers.has_entry(layer_name):
        doc.layers.new(layer_name, dxfattribs={"linetype": "CONTINUOUS"})
    msp = doc.modelspace()

    for dim in linear_dims:
        override = _dimension_overrides(dim.settings)
        entity = msp.add_linear_dim(
            base=dim.base,
            p1=dim.p1,
            p2=dim.p2,
            angle=dim.angle,
            override=override,
            dxfattribs={"layer": layer_name},
        )
        if dim.label:
            _set_dimension_text(entity, dim.label)
        entity.render()

    for dim in diameter_dims:
        override = _dimension_overrides(dim.settings)
        entity = msp.add_diameter_dim(
            center=dim.center,
            radius=dim.radius,
            angle=dim.angle,
            override=override,
            dxfattribs={"layer": layer_name},
        )
        if dim.label:
            _set_dimension_text(entity, dim.label)
        entity.render()


def render_dxf_to_svg(
    doc: "ezdxf.document.Drawing",
    page_size_mm: tuple[float, float],
    margin_mm: float = 1.0,
) -> str:
    ctx = RenderContext(doc)
    backend = SVGBackend()
    config = Configuration(background_policy=BackgroundPolicy.WHITE)
    page = layout.Page(
        page_size_mm[0],
        page_size_mm[1],
        layout.Units.mm,
        margins=layout.Margins.all(margin_mm),
    )
    Frontend(ctx, backend, config=config).draw_layout(doc.modelspace(), finalize=True)
    return backend.get_string(page)
