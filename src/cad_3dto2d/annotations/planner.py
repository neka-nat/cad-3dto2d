from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from ..types import Point2D
from .dimensions import DimensionOrientation, DimensionSide
from .features import CirclePrimitive, FeatureCoordinates


class PlannedDimension(BaseModel):
    model_config = ConfigDict(frozen=True)

    p1: Point2D
    p2: Point2D
    orientation: DimensionOrientation
    side: DimensionSide
    label: str | None = None
    base_ref: float | None = None


class PlannedDiameterDimension(BaseModel):
    model_config = ConfigDict(frozen=True)

    center: Point2D
    radius: float
    leader_angle_deg: float
    label: str | None = None


class PlannedPitchDimension(BaseModel):
    model_config = ConfigDict(frozen=True)

    p1: Point2D
    p2: Point2D
    orientation: DimensionOrientation
    side: DimensionSide
    count: int
    pitch: float
    base_ref: float | None


class PlanningRules(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_line_dims: int = 8
    max_hole_positions: int = 4
    max_pitch_dims: int = 2
    max_internal_dims: int = 4
    max_diameter_dims: int = 2
    collapse_diameter_with_pitch: bool = True
    max_diameter_groups: int = 1
    prefer_bolt_circle_note: bool = True
    bolt_circle_min_count: int = 4
    bolt_circle_radius_tol_ratio: float = 0.02
    bolt_circle_angle_tol_deg: float = 2.0
    suppress_hole_dims_when_bolt_circle: bool = True


class BoltCirclePattern(BaseModel):
    model_config = ConfigDict(frozen=True)

    center: Point2D
    hole_radius: float
    count: int
    pcd_radius: float
    equal_spaced: bool


def _select_coords(coords: list[float], max_count: int) -> list[float]:
    if len(coords) <= max_count:
        return coords
    return [coords[0], coords[-1]]


def _group_circle_indices(
    circles: list[CirclePrimitive], axis: int, tol: float
) -> list[list[int]]:
    if not circles:
        return []
    indexed = list(enumerate(circles))
    indexed.sort(key=lambda item: item[1].center[axis])
    groups: list[list[int]] = [[indexed[0][0]]]
    group_ref = indexed[0][1].center[axis]
    for idx, circle in indexed[1:]:
        if abs(circle.center[axis] - group_ref) <= tol:
            groups[-1].append(idx)
        else:
            groups.append([idx])
            group_ref = circle.center[axis]
    return groups


def _has_close(value: float, values: list[float], tol: float) -> bool:
    return any(abs(value - other) <= tol for other in values)


def _detect_pitch_dimension(
    circles: list[CirclePrimitive],
    group_indices: list[int],
    axis: int,
    side: DimensionSide,
    min_pitch: float,
    pitch_tol_ratio: float,
    base_ref: float | None,
) -> PlannedPitchDimension | None:
    """Detect a pitch pattern along the given axis (0=x, 1=y)."""
    other_axis = 1 - axis
    orientation: DimensionOrientation = "horizontal" if axis == 0 else "vertical"
    sorted_indices = sorted(group_indices, key=lambda i: circles[i].center[axis])
    group_circles = [circles[i] for i in sorted_indices]
    coords = [c.center[axis] for c in group_circles]
    diffs = [
        coords[i + 1] - coords[i]
        for i in range(len(coords) - 1)
        if coords[i + 1] > coords[i]
    ]
    if not diffs:
        return None
    pitch = sum(diffs) / len(diffs)
    if pitch < min_pitch:
        return None
    if len(diffs) > 1 and (max(diffs) - min(diffs)) > pitch * pitch_tol_ratio:
        return None
    ref = group_circles[0].center[other_axis]
    if axis == 0:
        p1, p2 = (coords[0], ref), (coords[-1], ref)
    else:
        p1, p2 = (ref, coords[0]), (ref, coords[-1])
    return PlannedPitchDimension(
        p1=p1,
        p2=p2,
        orientation=orientation,
        side=side,
        count=len(group_indices),
        pitch=pitch,
        base_ref=base_ref,
    )


def plan_internal_dimensions(
    features: FeatureCoordinates,
    horizontal_side: DimensionSide,
    vertical_side: DimensionSide,
    max_per_axis: int = 2,
) -> list[PlannedDimension]:
    xmin, ymin, xmax, ymax = features.bounds
    dims: list[PlannedDimension] = []

    x_coords = _select_coords(sorted(features.x_coords), max_per_axis)
    y_coords = _select_coords(sorted(features.y_coords), max_per_axis)

    y_ref = ymax if horizontal_side == "top" else ymin
    for x in x_coords:
        dims.append(
            PlannedDimension(
                p1=(xmin, y_ref),
                p2=(x, y_ref),
                orientation="horizontal",
                side=horizontal_side,
            )
        )

    x_ref = xmax if vertical_side == "right" else xmin
    for y in y_coords:
        dims.append(
            PlannedDimension(
                p1=(x_ref, ymin),
                p2=(x_ref, y),
                orientation="vertical",
                side=vertical_side,
            )
        )

    return dims


def plan_hole_dimensions(
    features: FeatureCoordinates,
    horizontal_side: DimensionSide,
    vertical_side: DimensionSide,
    max_circles: int = 2,
    min_pitch_count: int = 3,
    pitch_tol_ratio: float = 0.1,
    min_pitch: float = 0.5,
) -> tuple[
    list[PlannedDimension],
    list[PlannedDiameterDimension],
    list[PlannedPitchDimension],
    dict[str, bool],
]:
    xmin, ymin, xmax, ymax = features.bounds
    y_ref = ymax if horizontal_side == "top" else ymin
    x_ref = xmax if vertical_side == "right" else xmin
    position_dims: list[PlannedDimension] = []
    diameter_dims: list[PlannedDiameterDimension] = []
    pitch_dims: list[PlannedPitchDimension] = []
    pitch_axes = {"horizontal": False, "vertical": False}

    circles = features.circles
    if circles:
        angle_candidates = [45, 135] if horizontal_side == "top" else [-45, -135]
        for idx, circle in enumerate(circles[:max_circles]):
            angle = angle_candidates[idx % len(angle_candidates)]
            diameter_dims.append(
                PlannedDiameterDimension(
                    center=circle.center, radius=circle.radius, leader_angle_deg=angle
                )
            )
        tol = 1e-3
        horizontal_groups = _group_circle_indices(circles, axis=1, tol=tol)
        vertical_groups = _group_circle_indices(circles, axis=0, tol=tol)
        skip_horizontal: set[int] = set()
        skip_vertical: set[int] = set()
        for group in horizontal_groups:
            if len(group) < min_pitch_count:
                continue
            pitch_dim = _detect_pitch_dimension(
                circles,
                group,
                axis=0,
                side=horizontal_side,
                min_pitch=min_pitch,
                pitch_tol_ratio=pitch_tol_ratio,
                base_ref=y_ref,
            )
            if pitch_dim:
                skip_horizontal.update(group)
                pitch_dims.append(pitch_dim)
                pitch_axes["horizontal"] = True
        for group in vertical_groups:
            if len(group) < min_pitch_count:
                continue
            pitch_dim = _detect_pitch_dimension(
                circles,
                group,
                axis=1,
                side=vertical_side,
                min_pitch=min_pitch,
                pitch_tol_ratio=pitch_tol_ratio,
                base_ref=x_ref,
            )
            if pitch_dim:
                skip_vertical.update(group)
                pitch_dims.append(pitch_dim)
                pitch_axes["vertical"] = True
        used_x: list[float] = []
        used_y: list[float] = []
        for idx, circle in enumerate(circles):
            cx, cy = circle.center
            if idx not in skip_horizontal and not _has_close(cx, used_x, tol):
                used_x.append(cx)
                position_dims.append(
                    PlannedDimension(
                        p1=(xmin, cy),
                        p2=(cx, cy),
                        orientation="horizontal",
                        side=horizontal_side,
                        base_ref=y_ref,
                    )
                )
            if idx not in skip_vertical and not _has_close(cy, used_y, tol):
                used_y.append(cy)
                position_dims.append(
                    PlannedDimension(
                        p1=(cx, ymin),
                        p2=(cx, cy),
                        orientation="vertical",
                        side=vertical_side,
                        base_ref=x_ref,
                    )
                )
        if pitch_axes["horizontal"] and pitch_axes["vertical"] and not position_dims:
            first = circles[0]
            position_dims.append(
                PlannedDimension(
                    p1=(xmin, first.center[1]),
                    p2=(first.center[0], first.center[1]),
                    orientation="horizontal",
                    side=horizontal_side,
                )
            )
    return position_dims, diameter_dims, pitch_dims, pitch_axes


def group_circles_by_radius(
    circles: list[CirclePrimitive], tol: float = 1e-3
) -> list[list[CirclePrimitive]]:
    if not circles:
        return []
    circles_sorted = sorted(circles, key=lambda c: c.radius)
    groups: list[list[CirclePrimitive]] = [[circles_sorted[0]]]
    for circle in circles_sorted[1:]:
        if abs(circle.radius - groups[-1][-1].radius) <= tol:
            groups[-1].append(circle)
        else:
            groups.append([circle])
    return groups


def apply_planning_rules(
    hole_positions: list[PlannedDimension],
    hole_pitches: list[PlannedDimension],
    internal_dims: list[PlannedDimension],
    hole_diameters: list[PlannedDiameterDimension],
    rules: PlanningRules | None = None,
) -> tuple[list[PlannedDimension], list[PlannedDiameterDimension]]:
    rules = rules or PlanningRules()
    line_dims: list[PlannedDimension] = []
    line_dims.extend(hole_positions[: rules.max_hole_positions])
    line_dims.extend(hole_pitches[: rules.max_pitch_dims])
    remaining = rules.max_line_dims - len(line_dims)
    if remaining > 0:
        line_dims.extend(internal_dims[: min(remaining, rules.max_internal_dims)])
    diameter_dims = hole_diameters[: rules.max_diameter_dims]
    return line_dims, diameter_dims


def _distance(a: Point2D, b: Point2D) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def _angles_deg(points: list[Point2D], center: Point2D) -> list[float]:
    out: list[float] = []
    for p in points:
        out.append(math.degrees(math.atan2(p[1] - center[1], p[0] - center[0])) % 360.0)
    return out


def detect_bolt_circle_pattern(
    circles: list[CirclePrimitive],
    bounds: tuple[float, float, float, float],
    min_count: int = 4,
    radius_tol_ratio: float = 0.02,  # 半径ばらつき許容(相対)
    angle_tol_deg: float = 2.0,  # 等配許容(度)
) -> BoltCirclePattern | None:
    if not circles:
        return None

    xmin, ymin, xmax, ymax = bounds
    center = ((xmin + xmax) / 2, (ymin + ymax) / 2)

    # 半径（穴径）ごとにグループ化
    groups = group_circles_by_radius(circles, tol=1e-3)

    best: BoltCirclePattern | None = None

    for group in groups:
        if len(group) < min_count:
            continue

        pts = [c.center for c in group]
        radii = [_distance(p, center) for p in pts]
        mean_r = sum(radii) / len(radii)

        if mean_r <= 1e-6:
            continue

        tol_r = max(1e-3, mean_r * radius_tol_ratio)
        if max(abs(r - mean_r) for r in radii) > tol_r:
            continue

        # 等配チェック（簡易）
        ang = sorted(_angles_deg(pts, center))
        n = len(ang)
        step = 360.0 / n
        diffs = [(ang[(i + 1) % n] - ang[i]) % 360.0 for i in range(n)]
        equal_spaced = max(abs(d - step) for d in diffs) <= angle_tol_deg

        candidate = BoltCirclePattern(
            center=center,
            hole_radius=group[0].radius,
            count=len(group),
            pcd_radius=mean_r,
            equal_spaced=equal_spaced,
        )

        # “最も穴数が多い”→同数なら“PCDが大きい”を優先
        if best is None:
            best = candidate
        else:
            if (candidate.count, candidate.pcd_radius) > (best.count, best.pcd_radius):
                best = candidate

    return best
