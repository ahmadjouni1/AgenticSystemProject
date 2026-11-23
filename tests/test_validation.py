from __future__ import annotations

import math

import pytest

from examples import four_bar, gantry
from sdk import Assembly, Box, LoopClosureError, Mimic, PoseError, ValidationError


def two_bodies(name: str = "a") -> tuple[Assembly, object, object]:
    asm = Assembly(name)
    base = asm.body("base")
    base.add("box", Box((0.1, 0.1, 0.1)))
    arm = asm.body("arm")
    arm.add("box", Box((0.1, 0.02, 0.02)))
    return asm, base, arm


# ---- names --------------------------------------------------------------------


def test_duplicate_body_name() -> None:
    asm, _, _ = two_bodies()
    with pytest.raises(ValidationError, match="duplicate body name 'base'"):
        asm.body("base")


def test_duplicate_joint_name() -> None:
    asm, base, arm = two_bodies()
    asm.joint("j", "fixed", base.at(), arm.at())
    with pytest.raises(ValidationError, match="duplicate joint name 'j'"):
        asm.joint("j", "fixed", base.at(), arm.at())


def test_duplicate_shape_name() -> None:
    asm, base, _ = two_bodies()
    with pytest.raises(ValidationError, match="already has a shape named 'box'"):
        base.add("box", Box((1, 1, 1)))


@pytest.mark.parametrize("bad", ["", "has space", "1st", "dash-name", None])
def test_invalid_names(bad: object) -> None:
    with pytest.raises(ValidationError, match="name must be"):
        Assembly("ok").body(bad)  # type: ignore[arg-type]


# ---- joints ---------------------------------------------------------------------


def test_lower_limit_above_upper() -> None:
    asm, base, arm = two_bodies()
    with pytest.raises(ValidationError, match="lower limit 1 exceeds upper 0.5"):
        asm.joint("j", "revolute", base.at(), arm.at(), axis=(0, 0, 1), limits=(1.0, 0.5))


@pytest.mark.parametrize("limits", [(0.2, 1.0), (-1.0, -0.1)])
def test_limits_must_contain_zero(limits: tuple[float, float]) -> None:
    asm, base, arm = two_bodies()
    with pytest.raises(ValidationError, match="must contain 0, the authored rest pose"):
        asm.joint("j", "revolute", base.at(), arm.at(), axis=(0, 0, 1), limits=limits)


def test_undriven_joint_sits_at_the_authored_pose() -> None:
    asm, base, arm = two_bodies()
    asm.joint("j", "revolute", base.at(), arm.at(), axis=(0, 0, 1), limits=(0.0, 1.0))
    assert asm.solve().joint_values["j"] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"type": "revolute"}, "needs an axis"),
        ({"type": "prismatic", "axis": (0, 0, 0)}, "non-zero vector"),
        ({"type": "fixed", "axis": (0, 0, 1)}, "takes no axis"),
        ({"type": "slider", "axis": (0, 0, 1)}, "type must be one of"),
        ({"type": "revolute", "axis": (0, 0, 1), "limits": (0, math.inf)}, "finite"),
    ],
)
def test_bad_joint_fields(kwargs: dict, message: str) -> None:
    asm, base, arm = two_bodies()
    joint_type = kwargs.pop("type")
    with pytest.raises(ValidationError, match=message):
        asm.joint("j", joint_type, base.at(), arm.at(), **kwargs)


def test_joint_to_itself() -> None:
    asm, base, _ = two_bodies()
    with pytest.raises(ValidationError, match="to itself"):
        asm.joint("j", "fixed", base.at(), base.at((1, 0, 0)))


def test_joint_with_foreign_body() -> None:
    asm, base, _ = two_bodies("first")
    _, _, stranger = two_bodies("second")
    with pytest.raises(ValidationError, match="not in this assembly"):
        asm.joint("j", "fixed", base.at(), stranger.at())


# ---- tree structure ---------------------------------------------------------------


def test_unconnected_body_means_two_roots() -> None:
    asm, _, _ = two_bodies()
    with pytest.raises(ValidationError, match=r"exactly one root, found 2: \['base', 'arm'\]"):
        asm.validate()


def test_body_with_two_parents() -> None:
    asm = Assembly("two_parents")
    left, right, shared = (asm.body(name) for name in ("left", "right", "shared"))
    for body in (left, right, shared):
        body.add("box", Box((0.1, 0.1, 0.1)))
    asm.joint("a", "fixed", left.at(), shared.at())
    asm.joint("b", "fixed", right.at(), shared.at())
    with pytest.raises(ValidationError, match="'shared' is the child of both 'a' and 'b'"):
        asm.validate()


def test_explicit_root_must_have_no_parent() -> None:
    asm, base, arm = two_bodies()
    asm.joint("j", "fixed", base.at(), arm.at())
    with pytest.raises(ValidationError, match="root 'arm' is the child of joint 'j'"):
        asm.articulation(root="arm")


def test_loop_without_closure_is_rejected() -> None:
    linkage = four_bar.build()
    with pytest.raises(ValidationError, match="tree joints form a loop"):
        linkage.articulation(root="ground", closures=[])


def test_closures_are_found_automatically() -> None:
    linkage = four_bar.build()
    art = linkage.articulation()  # no names given: the last joint of the ring closes it
    assert [j.name for j in art.closures] == ["pin"]
    assert art.root == "ground"
    assert [j.name for j in art.tree] == ["crank_joint", "rocker_joint", "coupler_joint"]


def test_unknown_closure_name() -> None:
    with pytest.raises(ValidationError, match="unknown joint 'nope'"):
        four_bar.build().articulation(closures=["nope"])


# ---- mimic --------------------------------------------------------------------------


def chain(*mimics: tuple[str, Mimic | None]) -> Assembly:
    """base -> b0 -> b1 ... with one revolute joint per entry."""
    asm = Assembly("chain")
    parent = asm.body("base")
    parent.add("box", Box((0.1, 0.1, 0.1)))
    for index, (name, mimic) in enumerate(mimics):
        child = asm.body(f"b{index}")
        child.add("bar", Box((0.1, 0.02, 0.02)))
        asm.joint(name, "revolute", parent.at((0.1, 0, 0)), child.at(), axis=(0, 0, 1),
                  mimic=mimic)
        parent = child
    return asm


def test_mimic_source_must_exist() -> None:
    with pytest.raises(ValidationError, match="mimics 'ghost', which does not exist"):
        chain(("a", None), ("b", Mimic("ghost"))).validate()


def test_mimic_cycle() -> None:
    asm = chain(("a", Mimic("c")), ("b", Mimic("a")), ("c", Mimic("b")))
    with pytest.raises(ValidationError, match="mimic cycle: a -> c -> b -> a"):
        asm.validate()


def test_mimic_of_itself() -> None:
    with pytest.raises(ValidationError, match="cannot mimic itself"):
        chain(("a", Mimic("a")))


def test_mimic_of_fixed_joint() -> None:
    asm, base, arm = two_bodies()
    tip = asm.body("tip")
    tip.add("box", Box((0.01, 0.01, 0.01)))
    asm.joint("weld", "fixed", base.at(), arm.at())
    asm.joint("spin", "revolute", arm.at(), tip.at(), axis=(0, 0, 1), mimic=("weld", 1.0, 0.0))
    with pytest.raises(ValidationError, match="mimics fixed joint 'weld'"):
        asm.validate()


def test_zero_multiplier() -> None:
    with pytest.raises(ValidationError, match="non-zero"):
        Mimic("a", 0.0)


def slides(source_limits, follower_limits, mimic) -> Assembly:
    asm = Assembly("slides")
    base, a, b = (asm.body(name) for name in ("base", "a", "b"))
    for body in (base, a, b):
        body.add("box", Box((0.05, 0.05, 0.05)))
    asm.joint("ja", "prismatic", base.at(), a.at(), axis=(1, 0, 0), limits=source_limits)
    asm.joint("jb", "prismatic", a.at(), b.at(), axis=(1, 0, 0), limits=follower_limits,
              mimic=mimic)
    return asm


def test_mimic_exceeding_source_range_fails_at_validation() -> None:
    # ja reaches 1.0, so jb = 1.0 * ja would reach 1.0, beyond its 0.5 limit.
    with pytest.raises(ValidationError, match=r"'jb' reaches \(0, 1\) over the range of 'ja'"):
        slides((0, 1), (0, 0.5), ("ja", 1.0, 0.0)).validate()
    # Halving fits: 0.5 * ja stays within (0, 0.5). A negative multiplier flips the range.
    slides((0, 1), (0, 0.5), ("ja", 0.5, 0.0)).validate()
    slides((0, 1), (-0.5, 0), ("ja", -0.5, 0.0)).validate()


def test_mimic_with_limits_needs_a_limited_source() -> None:
    with pytest.raises(ValidationError, match="its source 'ja' is unlimited"):
        slides(None, (0, 0.5), ("ja", 1.0, 0.0)).validate()


def test_mimic_chain_ranges_compound() -> None:
    asm = slides((0, 0.2), None, ("ja", 2.0, 0.0))  # jb reaches 0.4
    c = asm.body("c")
    c.add("box", Box((0.05, 0.05, 0.05)))
    asm.joint("jc", "prismatic", asm.get_body("b").at(), c.at(), axis=(1, 0, 0),
              limits=(0, 0.3), mimic=("jb", 1.0, 0.0))
    with pytest.raises(ValidationError, match=r"'jc' reaches \(0, 0.4\)"):
        asm.validate()


# ---- empty bodies, open loops at rest, frame_in ------------------------------------------


def test_body_without_shapes_fails() -> None:
    asm, base, _ = two_bodies()
    hollow = asm.body("hollow")
    asm.joint("j", "fixed", base.at(), hollow.at())
    with pytest.raises(ValidationError, match=r"bodies with no shapes: \['hollow'\]"):
        asm.validate()


def test_loop_open_at_rest_fails_with_clear_message() -> None:
    # A square of four bars, but the closing pin is put 5 mm off the coupler's end.
    asm = Assembly("open_ring")
    ground, crank, coupler, rocker = (asm.body(n) for n in ("ground", "crank", "coupler", "rocker"))
    for body in (ground, crank, coupler, rocker):
        body.add("bar", Box((0.1, 0.01, 0.01)), xyz=(0.05, 0, 0))
    z, up, right = (0, 0, 1), (0, 0, math.pi / 2), (0, 0, -math.pi / 2)
    # crank and rocker point up (+y) from x = 0 and x = 0.1; the coupler spans the top.
    asm.joint("crank_joint", "revolute", ground.at(rpy=up), crank.at(), axis=z)
    asm.joint("coupler_joint", "revolute", crank.at((0.1, 0, 0), rpy=right), coupler.at(),
              axis=z)
    asm.joint("rocker_joint", "revolute", ground.at((0.1, 0, 0), rpy=up), rocker.at(), axis=z)
    asm.joint("pin", "revolute", coupler.at((0.105, 0, 0)), rocker.at((0.1, 0, 0)), axis=z)
    with pytest.raises(ValidationError) as caught:
        asm.articulation(closures=["pin"])
    message = str(caught.value)
    assert "loop closure 'pin' is open in the rest pose" in message
    assert "5 mm apart" in message
    assert "frame_in" in message


def test_frame_in_needs_connected_bodies() -> None:
    asm, base, arm = two_bodies()
    with pytest.raises(ValidationError, match="not connected by joints yet"):
        asm.frame_in(base.at((0.1, 0, 0)), arm)


# ---- pose errors ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"nope": 0.0}, "unknown joint 'nope'"),
        ({"x_follow": 0.1}, "mimics 'x_drive'; drive that instead"),
        ({"bridge_mount": 0.0}, "is fixed"),
        ({"x_drive": 1.5}, "outside"),
        ({"x_drive": math.nan}, "finite"),
    ],
)
def test_bad_joint_values(values: dict, message: str) -> None:
    with pytest.raises(PoseError, match=message):
        gantry.build().solve(values)


def test_closure_cannot_be_driven() -> None:
    with pytest.raises(PoseError, match="closes a loop"):
        four_bar.build().solve({"pin": 0.1})


def test_loop_that_cannot_close_raises() -> None:
    # The rocker may only swing +-0.05 rad, but crank 180 deg needs about 0.68 rad.
    linkage = four_bar.build(rocker_limits=(-0.05, 0.05))
    with pytest.raises(LoopClosureError, match="stuck at a limit: \\['rocker_joint'\\]"):
        linkage.solve({"crank_joint": math.pi})
