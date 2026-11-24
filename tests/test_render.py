"""The CPU renderer: depth-correct drawing and the facing of the views (step 10 follow-up)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from agent.tools.render import DEFAULT_FRONT, front_of, rasterize, views, zbuffer


def _open(vertices) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)


def test_a_hidden_triangle_stays_hidden_where_sorting_by_centroid_fails() -> None:
    """The old painter's sort drew triangles by centroid depth. A large slanted triangle has
    its centroid far away, so a small triangle below its near corner was drawn over it: the
    dark wedges of the step 10 renders. The depth buffer hides it."""
    big = _open([[-1, -1, 0], [1, -1, 0], [0, 1, 2]])  # the surface is at z = y + 1
    small = _open([[-0.05, 0.8, 1.5], [0.05, 0.8, 1.5], [0, 0.9, 1.5]])  # 0.3-0.4 below it
    assert small.triangles.mean(axis=1)[0, 2] > big.triangles.mean(axis=1)[0, 2]
    face, _colors, owners, _scale = zbuffer([("big", big), ("small", small)],
                                    np.array([0.0, 0.0, -1.0]), 400, 400)
    seen = owners[face[face >= 0]]
    assert (seen == 0).sum() > 1000 and (seen == 1).sum() == 0


def test_a_box_renders_its_three_visible_faces_and_nothing_else() -> None:
    box = trimesh.creation.box((1, 1, 1))
    forward, right, _label = views()["iso"]
    face, colors, _owners, _scale = zbuffer([("box", box)], forward, 300, 300, right)
    shades = {tuple(colors[i].round(3)) for i in np.unique(face[face >= 0])}
    assert len(shades) == 3  # top, front, side: flat shading, no stray back faces


def test_the_default_views_are_the_old_fixed_directions() -> None:
    v = views()
    assert np.allclose(v["front"][0], [0, 1, 0]) and np.allclose(v["side"][0], [-1, 0, 0])
    assert np.allclose(v["top"][0], [0, 0, -1]) and np.allclose(v["top"][1], [1, 0, 0])
    assert np.allclose(v["iso"][0], np.array([-1, 1, -0.8]) / np.linalg.norm([1, 1, 0.8]))
    assert v["iso"][2] == "iso (from +X -Y +Z)"


@pytest.mark.parametrize("door_y,expected", [(0.52, [0, 1, 0]), (-0.52, [0, -1, 0])])
def test_the_views_face_the_side_with_the_moving_parts(door_y, expected) -> None:
    carcass = trimesh.creation.box((1.0, 1.0, 1.0))
    door = trimesh.creation.box((0.9, 0.04, 0.9)).apply_translation((0, door_y, 0))
    front = front_of([("carcass", carcass), ("door", door)], {"door"})
    assert np.allclose(front, expected)
    forward, _right, label = views(front)["iso"]
    assert forward @ front < 0  # the camera is on the front side
    assert _axis_in(label, expected)


def _axis_in(label: str, expected) -> bool:
    return ("+Y" if expected[1] > 0 else "-Y") in label


def test_no_moving_parts_keeps_the_default_front() -> None:
    box = trimesh.creation.box((1, 1, 1))
    assert np.allclose(front_of([("box", box)], set()), DEFAULT_FRONT)


def test_rasterize_keeps_its_size_and_labels() -> None:
    image = rasterize([("box", trimesh.creation.box((1, 1, 1)))], *views()["iso"][:1],
                      "iso", views()["iso"][1])
    assert image.size == (640, 480)
