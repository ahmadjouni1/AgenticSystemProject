"""A ready-to-compile skeleton ``main.py`` for a solved four-bar or slider-crank.

``skeleton_source(mechanism, result, target)`` writes plain SDK code: simple bars at the
solved pivots, the loop closed with ``frame_in`` (so the rest pose is closed by
construction), and ``target_path`` declared from the target. The model compiles it first,
then adds detail in small edits.

**Layers, so it passes the checks.** The links are bars stacked along the plane's normal;
links joined by a joint touch face to face around the joint, so they stay in contact while
they turn, and links that are not joined never share a layer where they could meet:

| Layer (thickness T) | four-bar | slider-crank |
| --- | --- | --- |
| below the plane | ground plate | ground plate |
| 1 | crank | crank, and the slider block (beside it, on the rail) |
| 2 | coupler | rod |
| 3 | rocker, on a pivot post from the plate | |

The rocker's post passes the crank and coupler layers at the rocker pivot. The crank clears
it when the ground link is longer than the crank (a crank-rocker), and the coupler clears it
by the rocker's length x sin(transmission angle), at least 0.57 x rocker at 35 degrees. The
slider block clears the crank when the rod is longer than twice the crank. Otherwise
``SkeletonError`` says why, and the tool reports it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sdk.transforms import rpy_from_matrix, rpy_matrix
from solvers.base import SolverResult
from solvers.planar import PLANES


class SkeletonError(Exception):
    pass


def _plane(plane: str, height: float) -> tuple[np.ndarray, np.ndarray]:
    """(rotation whose columns are the plane's u, v and normal, origin of the plane)."""
    u, v, normal = PLANES[plane]
    axes = np.eye(3)
    rotation = np.column_stack([axes[u], axes[v], np.cross(axes[u], axes[v])])
    origin = np.zeros(3)
    origin[normal] = height
    return rotation, origin


def _n(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0.0" if text in ("-0", "") else text


def _v(values) -> str:
    return "(" + ", ".join(_n(float(x)) for x in values) + ")"


def skeleton_source(mechanism: str, result: SolverResult, target: dict[str, Any]) -> str:
    p = result.parameters
    details = result.details
    plane = details.get("plane", "xy")
    height = details.get("height")
    if height is None:  # a path target: its points give the plane's height
        _u, _v_axis, normal = PLANES[plane]
        height = float(np.mean([point[normal] for point in target["points"]]))
    rotation, origin = _plane(plane, float(height))
    world = lambda x, y, z=0.0: origin + rotation @ np.array([x, y, z])  # noqa: E731
    turned = lambda angle: rpy_from_matrix(rotation @ rpy_matrix((0, 0, angle)))  # noqa: E731
    lengths = [p[k] for k in p if k.endswith("_length")]
    width = min(max(0.12 * min(lengths), 0.003), 0.02)
    thick = width / 2
    target_path = _target_path(mechanism, p, details, target)
    span = float(details.get("crank_span", 2 * math.pi))
    limits = "" if abs(abs(span) - 2 * math.pi) < 1e-9 else \
        f", limits=({_n(min(0.0, span))}, {_n(max(0.0, span))})"
    head = [f'"""{mechanism.replace("_", "-")} skeleton from synthesize: simple bars at the '
            'solved pivots. Add detail in small edits; keep the pivots and lengths."""',
            "", "import math", "", "from sdk import Assembly, Box, Cylinder", "",
            f"W, T = {_n(width)}, {_n(thick)}  # bar width and layer thickness (m)",
            f"PLANE = {_v(rpy_from_matrix(rotation))}  # rpy of the linkage's plane",
            f"UP = {_v(rotation[:, 2])}  # the plane's normal, the direction of the layers", ""]
    if mechanism == "four_bar":
        body = _four_bar(p, world, turned, width, thick, limits, details)
    else:
        body = _slider_crank(p, world, turned, width, thick, limits)
    return "\n".join([*head, *body, "", f"target_path = {target_path!r}", ""])


def _plate(world, centre: np.ndarray, size: np.ndarray, thick: float) -> str:
    middle = world(centre[0], centre[1], -thick / 2)
    return (f'ground.add("plate", Box(({_n(size[0])}, {_n(size[1])}, T)), '
            f"xyz={_v(middle)}, rpy=PLANE)")


def _four_bar(p, world, turned, width, thick, limits, details) -> list[str]:
    a, b, g = p["crank_length"], p["coupler_length"], p["ground_length"]
    crank_pivot = np.array([p["crank_pivot_x"], p["crank_pivot_y"]])
    rocker_pivot = crank_pivot + g * np.array([math.cos(p["ground_angle"]),
                                               math.sin(p["ground_angle"])])
    rocker = details.get("derived", {}).get("rocker_length", 0.0)
    if g - a <= width:
        raise SkeletonError("the crank reaches the rocker pivot (ground link not longer than "
                            "the crank by a bar width); build this linkage by hand")
    mu = details.get("derived", {}).get("min_transmission_deg")
    if mu is not None and rocker * math.sin(math.radians(mu)) <= width:
        raise SkeletonError("the coupler passes too close to the rocker pivot; build this "
                            "linkage by hand")
    corners = np.array([crank_pivot - (a + width), crank_pivot + (a + width),
                        rocker_pivot - width, rocker_pivot + width])
    low, high = corners.min(axis=0), corners.max(axis=0)
    post = world(*rocker_pivot, 1.5 * thick)
    return [
        'model = Assembly("four_bar")',
        f"CRANK, COUPLER = {_n(a)}, {_n(b)}  # m; the rocker follows from the closure",
        "",
        'ground = model.body("ground")',
        _plate(world, (low + high) / 2, high - low, thick),
        f'ground.add("pivot_post", Cylinder(radius=0.35 * W, height=3 * T), xyz={_v(post)}, '
        "rpy=PLANE)",
        'crank = model.body("crank")',
        'crank.add("bar", Box((CRANK + W, W, T)), xyz=(CRANK / 2, 0, T / 2))',
        'coupler = model.body("coupler")',
        'coupler.add("bar", Box((COUPLER + W, W, T)), xyz=(COUPLER / 2, 0, 1.5 * T))',
        'rocker = model.body("rocker")',
        "",
        f'model.joint("crank_joint", "revolute", ground.at({_v(world(*crank_pivot))}, '
        f'rpy={_v(turned(p["crank_phase"]))}), crank.at(), axis=(0, 0, 1){limits})',
        'model.joint("coupler_joint", "revolute", crank.at((CRANK, 0, 0), '
        f'rpy=(0, 0, {_n(p["coupler_angle"])})), coupler.at(), axis=(0, 0, 1))',
        f'model.joint("rocker_joint", "revolute", ground.at({_v(world(*rocker_pivot))}, '
        "rpy=PLANE), rocker.at(), axis=(0, 0, 1))",
        "tip = coupler.at((COUPLER, 0, 0))",
        "pin = model.frame_in(tip, rocker)  # where the rocker meets the coupler at rest",
        "x, y = pin.frame.xyz[0], pin.frame.xyz[1]",
        "ROCKER, ANGLE = math.hypot(x, y), math.atan2(y, x)",
        'rocker.add("bar", Box((ROCKER + W, W, T)), xyz=(x / 2, y / 2, 2.5 * T), '
        "rpy=(0, 0, ANGLE))",
        'model.joint("pin", "revolute", tip, pin, axis=(0, 0, 1))',
        'model.articulation(root="ground", closures=["pin"])',
        "object_model = model",
        "",
        "",
        "def run_tests(ctx):",
        '    ctx.allow_overlap("ground", "rocker", reason="the rocker turns on the pivot post '
        'through its eye", shape_a="pivot_post", shape_b="bar")',
    ]


def _slider_crank(p, world, turned, width, thick, limits) -> list[str]:
    a, b = p["crank_length"], p["rod_length"]
    room = b - 2 * a - 1.5 * width
    if room < width:
        raise SkeletonError("the rod is too short for the slider block to clear the crank "
                            "(it needs more than twice the crank); build this linkage by hand")
    block = min(max(room / 2, width / 2), 2 * width)
    crank_pivot = np.array([p["crank_pivot_x"], p["crank_pivot_y"]])
    reach = a + b + 2 * width
    return [
        'model = Assembly("slider_crank")',
        f"CRANK, ROD, BLOCK = {_n(a)}, {_n(b)}, {_n(2 * block)}  # m",
        "",
        'ground = model.body("ground")',
        _plate(world, crank_pivot, np.array([2 * reach, 2 * reach]), thick),
        'crank = model.body("crank")',
        'crank.add("bar", Box((CRANK + W, W, T)), xyz=(CRANK / 2, 0, T / 2))',
        'rod = model.body("rod")',
        'rod.add("bar", Box((ROD + W, W, T)), xyz=(ROD / 2, 0, 1.5 * T))',
        'slider = model.body("slider")',
        'slider.add("block", Box((BLOCK, W, T)), xyz=(0, 0, T / 2))',
        "",
        f'model.joint("crank_joint", "revolute", ground.at({_v(world(*crank_pivot))}, '
        f'rpy={_v(turned(p["crank_phase"]))}), crank.at(), axis=(0, 0, 1){limits})',
        'model.joint("rod_joint", "revolute", crank.at((CRANK, 0, 0), '
        f'rpy=(0, 0, {_n(p["rod_angle"])})), rod.at(), axis=(0, 0, 1))',
        "tip = rod.at((ROD, 0, 0))",
        "rail = model.frame_in(tip, ground).frame.xyz  # the rail passes the slider pin at rest",
        f'model.joint("slide", "prismatic", ground.at(rail, rpy={_v(turned(p["slider_angle"]))}),'
        " slider.at(), axis=(1, 0, 0))",
        'model.joint("pin", "revolute", tip, model.frame_in(tip, slider), axis=(0, 0, 1))',
        'model.articulation(root="ground", closures=["pin"])',
        "object_model = model",
    ]


def _target_path(mechanism: str, p: dict, details: dict, target: dict) -> dict[str, Any]:
    kind = details.get("kind")
    if kind in ("sweep", "stroke"):
        return {"joint": "crank_joint", "output_joint": details["output_joint"],
                kind: target[kind], "tolerance": details["tolerance"]}
    if kind == "function":
        return {"joint": "crank_joint", "output_joint": details["output_joint"],
                "values": target["crank"], "outputs": target["output"],
                "tolerance": details["tolerance"]}
    body = "coupler" if mechanism == "four_bar" else "rod"
    return {"body": body, "point": [round(p["point_x"], 6), round(p["point_y"], 6), 0.0],
            "joint": "crank_joint", "points": target["points"],
            "tolerance": details["tolerance"]}
