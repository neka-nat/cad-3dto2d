from __future__ import annotations

from typing import Iterable

from build123d import Compound
from pydantic import BaseModel, ConfigDict

from .types import Point3D, Shape


def bounding_size(shapes: Iterable[Shape]) -> Point3D:
    shapes = list(shapes)
    if not shapes:
        return (0.0, 0.0, 0.0)
    size = Compound(children=shapes).bounding_box().size
    return (size.X, size.Y, size.Z)


class _ShapeModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class ViewProjection(_ShapeModel):
    visible: list[Shape]
    hidden: list[Shape]

    def bounding_size(self) -> Point3D:
        return bounding_size(self.visible + self.hidden)


class ThreeViewProjections(_ShapeModel):
    front: ViewProjection
    side_x: ViewProjection
    side_y: ViewProjection


def project_view(model, view_port_org: Point3D) -> ViewProjection:
    visible, hidden = model.project_to_viewport(view_port_org)
    return ViewProjection(visible=list(visible), hidden=list(hidden))


def project_three_views(model) -> ThreeViewProjections:
    return ThreeViewProjections(
        front=project_view(model, (0, 0, 1000000)),
        side_x=project_view(model, (1000000, 0, 0)),
        side_y=project_view(model, (0, 1000000, 0)),
    )
