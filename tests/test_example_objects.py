"""Every example builds, validates, poses, and is made of closed meshes."""

from __future__ import annotations

import importlib

import pytest

EXAMPLES = {
    # module: (helper category, a pose within the limits)
    "laptop": ("primitives", {"hinge": 1.2}),
    "four_bar": ("loop closure + wire_from_points", {"crank_joint": 2.0}),
    "gantry": ("mimic", {"x_drive": 0.4, "y_head": 0.1}),
    "toaster": ("primitives + booleans", {"lever_slide": -0.08, "browning": 2.0}),
    "door_hinge": ("extrusions + body.at(face)", {"pin": 1.5}),
    "desk_lamp": ("revolve + tubes", {"swivel": 1.0, "tilt": -0.3}),
    "computer_mouse": ("loft + profiles + sweep", {"left_button_click": -0.03, "scroll": 4.0}),
    "folding_chair": ("wires + pipe", {"fold": -0.5}),
}


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_example_builds_and_poses(name: str) -> None:
    _category, pose = EXAMPLES[name]
    model = importlib.import_module(f"examples.{name}").build()
    model.validate()
    transforms = model.forward_kinematics(pose)
    assert set(transforms) == {body.name for body in model.bodies}
    for body in model.bodies:
        for shape in body.shapes:
            mesh = shape.to_trimesh()
            assert len(mesh.faces) > 0
            assert mesh.is_watertight, f"{name}: {body.name}/{shape.name} is not closed"
