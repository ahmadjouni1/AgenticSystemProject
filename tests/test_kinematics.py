from __future__ import annotations

import math

import numpy as np
import pytest

from examples import four_bar, gantry, laptop
from sdk import Assembly, Box, LoopClosureError, Mimic, PoseError
from sdk.transforms import apply

# ---- laptop: one hinge, checked against hand-computed corner positions -----------------


def expected_lid_corner(q: float) -> tuple[float, float, float]:
    """Front-right top corner of the lid, by hand.

    In lid coordinates the corner is (W/2, -D, T). The hinge turns by q about -x,
    so y' = y cos q + z sin q and z' = -y sin q + z cos q. The hinge line is at
    (0, D/2, base thickness) in the world.
    """
    w, d, t = laptop.WIDTH, laptop.DEPTH, laptop.LID_THICKNESS
    y = -d * math.cos(q) + t * math.sin(q)
    z = d * math.sin(q) + t * math.cos(q)
    return (w / 2, d / 2 + y, laptop.BASE_THICKNESS + z)


@pytest.mark.parametrize("degrees", [0, 30, 45, 90, 120, 135])
def test_laptop_lid_corner(degrees: float) -> None:
    model = laptop.build()
    q = math.radians(degrees)
    lid = model.forward_kinematics({"hinge": q})["lid"]
    corner_local = (laptop.WIDTH / 2, -laptop.DEPTH, laptop.LID_THICKNESS)
    corner = apply(lid, corner_local)
    assert corner == pytest.approx(expected_lid_corner(q), abs=1e-12)

    # The lid mesh, moved by the same transform, really has a vertex at that corner.
    mesh = model.get_body("lid").shape("panel").to_trimesh()
    mesh.apply_transform(lid)
    assert np.min(np.linalg.norm(mesh.vertices - corner, axis=1)) < 1e-12


def test_laptop_known_values() -> None:
    model = laptop.build()
    closed = apply(model.forward_kinematics({"hinge": 0.0})["lid"], (0.16, -0.22, 0.008))
    upright = apply(model.forward_kinematics({"hinge": math.pi / 2})["lid"], (0.16, -0.22, 0.008))
    assert closed == pytest.approx((0.16, -0.11, 0.023))  # lying on the base
    assert upright == pytest.approx((0.16, 0.118, 0.235))  # straight up from the hinge


def test_laptop_base_stays_put_and_limits_hold() -> None:
    model = laptop.build()
    assert np.allclose(model.forward_kinematics({"hinge": 1.0})["base"], np.eye(4))
    for degrees in (-1, 136):
        with pytest.raises(PoseError, match="outside"):
            model.forward_kinematics({"hinge": math.radians(degrees)})


# ---- four-bar: the loop stays closed over a full crank turn ---------------------------------


ROCKER = four_bar.rocker_length(four_bar.build())  # derived by frame_in: about 70 mm


def rocker_tip(pose) -> np.ndarray:
    return apply(pose.transforms["rocker"], four_bar.build().get_joint("pin").child.frame.xyz)


def analytic_pin(theta: float) -> np.ndarray:
    """Coupler-rocker pin by circle intersection, on the branch of the rest pose."""
    a, b, c, d = four_bar.CRANK, four_bar.COUPLER, ROCKER, four_bar.GROUND
    crank_tip = np.array([a * math.cos(theta), a * math.sin(theta)])
    to_rocker = np.array([d, 0.0]) - crank_tip
    e = np.linalg.norm(to_rocker)
    u = to_rocker / e
    left = np.array([-u[1], u[0]])  # the rest pose has the pin left of crank tip -> rocker pivot
    along = (b**2 - c**2 + e**2) / (2 * e)
    return crank_tip + along * u + math.sqrt(b**2 - along**2) * left


def test_four_bar_closes_over_full_turn() -> None:
    linkage = four_bar.build()
    rocker_angles = []
    for theta in np.linspace(0.0, 2 * math.pi, 73):  # 5-degree steps
        pose = linkage.solve({"crank_joint": float(theta)})
        coupler_tip = apply(pose.transforms["coupler"], (four_bar.COUPLER, 0, 0))
        assert pose.residual < 1e-9
        assert coupler_tip == pytest.approx(rocker_tip(pose), abs=1e-9)
        assert coupler_tip[:2] == pytest.approx(analytic_pin(float(theta)), abs=1e-9)
        assert coupler_tip[2] == pytest.approx(0.0, abs=1e-12)  # stays planar
        rocker_angles.append(pose.joint_values["rocker_joint"])
    # Crank-rocker: the rocker swings and comes back, it never turns over.
    assert rocker_angles[0] == pytest.approx(0.0, abs=1e-9)
    assert rocker_angles[-1] == pytest.approx(0.0, abs=1e-9)
    assert max(rocker_angles) - min(rocker_angles) < math.pi / 2


def test_four_bar_links_keep_their_length() -> None:
    pose = four_bar.build().solve({"crank_joint": 2.0})
    ground_pivot = np.array([four_bar.GROUND, 0, 0])
    assert np.linalg.norm(rocker_tip(pose) - ground_pivot) == pytest.approx(ROCKER)
    assert ROCKER == pytest.approx(0.0700, abs=1e-4)  # a Grashof crank-rocker


def test_four_bar_rest_pose_is_closed_by_frame_in() -> None:
    # frame_in put the pin exactly on the coupler tip, so validate() finds no gap.
    linkage = four_bar.build()
    pose = linkage.solve()
    assert pose.residual < 1e-12
    pin = linkage.get_joint("pin")
    on_coupler = apply(pose.transforms["coupler"], pin.parent.frame.xyz)
    assert on_coupler == pytest.approx(rocker_tip(pose), abs=1e-12)


def test_big_jumps_stay_on_the_rest_branch() -> None:
    # A plain least-squares solve from rest straight to 180 deg lands "elbow down"
    # (rocker near 121 deg). Walking from the last pose keeps it "elbow up".
    for sequence in ([math.pi], [math.pi, 0.2, 4.5, 1.0], [3 * math.pi / 2, math.pi / 2]):
        linkage = four_bar.build()
        for theta in sequence:
            pose = linkage.solve({"crank_joint": theta})
            pin = apply(pose.transforms["coupler"], (four_bar.COUPLER, 0, 0))[:2]
            assert pin == pytest.approx(analytic_pin(theta), abs=1e-9)


def test_warm_start_keeps_branch_and_can_be_overridden() -> None:
    linkage = four_bar.build()
    for theta in np.linspace(0.0, math.pi, 19):  # walk to 180 degrees
        linkage.solve({"crank_joint": float(theta)})
    walked = linkage.solve({"crank_joint": math.pi}).joint_values["rocker_joint"]

    seeded = four_bar.build().solve(
        {"crank_joint": math.pi},
        warm_start={"coupler_joint": -0.6, "rocker_joint": 0.68},
    )
    assert seeded.joint_values["rocker_joint"] == pytest.approx(walked, abs=1e-9)


# ---- mimic ---------------------------------------------------------------------------------


@pytest.mark.parametrize("drive", [0.0, 0.25, 0.9])
def test_gantry_mimics_follow_source(drive: float) -> None:
    pose = gantry.build().solve({"x_drive": drive, "y_head": -0.2})
    values = pose.joint_values
    assert values["x_follow"] == pytest.approx(drive)
    assert values["cable_trolley"] == pytest.approx(0.5 * drive)
    assert pose.residual < 1e-12  # the bridge bolts onto the right carriage exactly

    left, right = pose.transforms["left_carriage"], pose.transforms["right_carriage"]
    assert left[0, 3] == pytest.approx(gantry.START_X + drive)
    assert right[0, 3] == pytest.approx(left[0, 3])
    assert pose.transforms["trolley"][0, 3] == pytest.approx(gantry.START_X + 0.5 * drive)
    assert pose.transforms["tool_head"][1, 3] == pytest.approx(-0.2)


def test_gantry_wrong_mimic_breaks_the_loop() -> None:
    model = gantry.build(follow_multiplier=0.9)
    model.solve({"x_drive": 0.0})  # equal at zero, so this still closes
    with pytest.raises(LoopClosureError, match="no free joint is left"):
        model.solve({"x_drive": 0.5})


def test_mimic_chain_with_offset() -> None:
    asm = Assembly("chain")
    base, a, b, c = (asm.body(name) for name in ("base", "a", "b", "c"))
    for body in (base, a, b, c):
        body.add("bar", Box((0.1, 0.01, 0.01)), xyz=(0.05, 0, 0))
    z = (0, 0, 1)
    asm.joint("ja", "revolute", base.at(), a.at(), axis=z)
    asm.joint("jb", "revolute", a.at((0.1, 0, 0)), b.at(), axis=z, mimic=("ja", -2.0, 0.1))
    asm.joint("jc", "revolute", b.at((0.1, 0, 0)), c.at(), axis=z, mimic=Mimic("jb", 0.5))

    pose = asm.solve({"ja": 0.3})
    assert pose.joint_values["jb"] == pytest.approx(-2.0 * 0.3 + 0.1)
    assert pose.joint_values["jc"] == pytest.approx(0.5 * (-2.0 * 0.3 + 0.1))
    # b's world heading is ja + jb; c's is ja + jb + jc.
    heading = lambda t: math.atan2(t[1, 0], t[0, 0])  # noqa: E731
    assert heading(pose.transforms["b"]) == pytest.approx(0.3 - 0.5)
    assert heading(pose.transforms["c"]) == pytest.approx(0.3 - 0.5 - 0.25)


def test_mimic_offset_sets_the_rest_value() -> None:
    # With the source at 0, a mimic sits at its offset; frame_in sees the same pose.
    asm = Assembly("offset")
    base, a, b = (asm.body(name) for name in ("base", "a", "b"))
    for body in (base, a, b):
        body.add("box", Box((0.02, 0.02, 0.02)))
    asm.joint("ja", "prismatic", base.at(), a.at(), axis=(1, 0, 0), limits=(-0.1, 0.1))
    asm.joint("jb", "prismatic", base.at((0, 0.1, 0)), b.at(), axis=(1, 0, 0),
              limits=(-0.2, 0.2), mimic=("ja", 1.0, 0.05))
    pose = asm.solve()
    assert pose.joint_values["jb"] == pytest.approx(0.05)
    assert pose.transforms["b"][0, 3] == pytest.approx(0.05)
    assert asm.frame_in(b.at(), base).frame.xyz == pytest.approx((0.05, 0.1, 0))
