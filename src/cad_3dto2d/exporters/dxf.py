from __future__ import annotations

from build123d import ExportDXF, LineType, Unit, Wire, Face, Edge

Shape = Wire | Face | Edge

_LINE_TYPES: dict[str, LineType] = {
    "visible": LineType.CONTINUOUS,
    "hidden": LineType.DASHED,
    "template": LineType.CONTINUOUS,
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
