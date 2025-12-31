from __future__ import annotations

from typing import Iterable

from build123d import Compound, Line
from pydantic import BaseModel, ConfigDict

from ..views import Shape


class DimensionSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    offset: float
    extension_gap: float
    arrow_size: float
    text_height: float = 3.5
    text_gap: float = 1.0
    decimal_places: int = 1


def _default_settings(size_x: float, size_y: float) -> DimensionSettings:
    size_ref = max(size_x, size_y, 1.0)
    return DimensionSettings(
        offset=max(5.0, size_ref * 0.1),
        extension_gap=max(0.5, size_ref * 0.02),
        arrow_size=max(2.0, size_ref * 0.03),
        text_height=max(2.5, size_ref * 0.04),
        text_gap=max(1.0, size_ref * 0.02),
        decimal_places=1,
    )


def _maybe_line(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> Line | None:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    if (dx * dx + dy * dy + dz * dz) <= 1e-12:
        return None
    return Line(p1, p2)


def default_settings_from_size(size_x: float, size_y: float) -> DimensionSettings:
    return _default_settings(size_x, size_y)


class DimensionText(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    text: str
    height: float
    anchor: str = "middle"


class DimensionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    lines: list[Line]
    texts: list[DimensionText]


def _format_length(value: float, decimal_places: int) -> str:
    if decimal_places <= 0:
        return str(int(round(value)))
    text = f"{value:.{decimal_places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _arrow_head_horizontal(x: float, y: float, arrow_size: float, direction: float) -> list[Line]:
    half = arrow_size / 2
    return [
        Line((x, y, 0), (x + direction * arrow_size, y + half, 0)),
        Line((x, y, 0), (x + direction * arrow_size, y - half, 0)),
    ]


def _arrow_head_vertical(x: float, y: float, arrow_size: float, direction: float) -> list[Line]:
    half = arrow_size / 2
    return [
        Line((x, y, 0), (x + half, y + direction * arrow_size, 0)),
        Line((x, y, 0), (x - half, y + direction * arrow_size, 0)),
    ]


def _horizontal_dimension(
    x1: float,
    x2: float,
    base_y: float,
    offset: float,
    settings: DimensionSettings,
) -> list[Line]:
    if x2 < x1:
        x1, x2 = x2, x1
    gap = min(settings.extension_gap, abs(offset)) * (1.0 if offset >= 0 else -1.0)
    dim_y = base_y + offset
    ext_start = base_y + gap
    lines: list[Line] = [
        line
        for line in (
            _maybe_line((x1, ext_start, 0), (x1, dim_y, 0)),
            _maybe_line((x2, ext_start, 0), (x2, dim_y, 0)),
            _maybe_line((x1, dim_y, 0), (x2, dim_y, 0)),
        )
        if line
    ]
    lines.extend(_arrow_head_horizontal(x1, dim_y, settings.arrow_size, direction=1.0))
    lines.extend(_arrow_head_horizontal(x2, dim_y, settings.arrow_size, direction=-1.0))
    return lines


def _vertical_dimension(
    y1: float,
    y2: float,
    base_x: float,
    offset: float,
    settings: DimensionSettings,
) -> list[Line]:
    if y2 < y1:
        y1, y2 = y2, y1
    gap = min(settings.extension_gap, abs(offset)) * (1.0 if offset >= 0 else -1.0)
    dim_x = base_x + offset
    ext_start = base_x + gap
    lines: list[Line] = [
        line
        for line in (
            _maybe_line((ext_start, y1, 0), (dim_x, y1, 0)),
            _maybe_line((ext_start, y2, 0), (dim_x, y2, 0)),
            _maybe_line((dim_x, y1, 0), (dim_x, y2, 0)),
        )
        if line
    ]
    lines.extend(_arrow_head_vertical(dim_x, y1, settings.arrow_size, direction=1.0))
    lines.extend(_arrow_head_vertical(dim_x, y2, settings.arrow_size, direction=-1.0))
    return lines


def generate_basic_dimensions(
    shapes: Iterable[Shape],
    settings: DimensionSettings | None = None,
    horizontal_dir: int = 1,
    vertical_dir: int = 1,
    horizontal_offset: float | None = None,
    vertical_offset: float | None = None,
) -> DimensionResult:
    shapes = list(shapes)
    if not shapes:
        return DimensionResult(lines=[], texts=[])
    bounds = Compound(children=shapes).bounding_box()
    size = bounds.size
    settings = settings or _default_settings(size.X, size.Y)
    xmin, ymin = bounds.min.X, bounds.min.Y
    xmax, ymax = bounds.max.X, bounds.max.Y
    if horizontal_offset is None:
        horizontal_offset = settings.offset * (1.0 if horizontal_dir >= 0 else -1.0)
    if vertical_offset is None:
        vertical_offset = settings.offset * (1.0 if vertical_dir >= 0 else -1.0)
    base_y = ymax if horizontal_dir >= 0 else ymin
    base_x = xmax if vertical_dir >= 0 else xmin
    lines: list[Line] = []
    texts: list[DimensionText] = []

    dim_y = base_y + horizontal_offset
    dim_x = base_x + vertical_offset
    lines.extend(_horizontal_dimension(xmin, xmax, base_y, horizontal_offset, settings))
    lines.extend(_vertical_dimension(ymin, ymax, base_x, vertical_offset, settings))

    width_label = _format_length(xmax - xmin, settings.decimal_places)
    height_label = _format_length(ymax - ymin, settings.decimal_places)
    horizontal_sign = 1 if horizontal_dir >= 0 else -1
    vertical_sign = 1 if vertical_dir >= 0 else -1
    texts.append(
        DimensionText(
            x=(xmin + xmax) / 2,
            y=dim_y + horizontal_sign * settings.text_gap,
            text=width_label,
            height=settings.text_height,
            anchor="middle",
        )
    )
    texts.append(
        DimensionText(
            x=dim_x + vertical_sign * settings.text_gap,
            y=(ymin + ymax) / 2,
            text=height_label,
            height=settings.text_height,
            anchor="start" if vertical_sign >= 0 else "end",
        )
    )
    return DimensionResult(lines=lines, texts=texts)
