"""render: the newest build at a pose, as PNG views, plus a text summary.

CPU only (numpy + Pillow): orthographic views, flat Lambert shading, back-face culling on
closed bodies, a per-pixel depth buffer, 2x supersampling. Colours match the viewer's
palette, which has no reds (red is kept for collisions).

**Facing.** The views turn with the object's front, which is the horizontal side (-Y, +Y, +X
or -X) from which the iso view shows the most of the moving bodies (doors, drawers, lids).
-Y is the default, kept unless another side shows clearly more. Every label states its real
direction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from fileio import atomic_write_bytes

from agent.tools.builds import (POSE_SCHEMA, Build, joint_report, newest_build, round_list,
                                solve_pose, world_meshes)
from agent.tools.core import AgentTool, ToolContext, ToolError
from sdk.errors import LoopClosureError, PoseError

WIDTH, HEIGHT, SUPERSAMPLE = 640, 480, 2
BACKGROUND = (243, 244, 246)
PALETTE = [0x4E79A7, 0xF28E2B, 0x76B7B2, 0x59A14F, 0xEDC948, 0xB07AA1, 0xB59B5A, 0xA0CBE8,
           0x8CD17D, 0xBAB0AC]  # = viewer/static/display.js
_Z = np.array([0.0, 0.0, 1.0])
DEFAULT_FRONT = np.array([0.0, -1.0, 0.0])
FRONTS = (DEFAULT_FRONT, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([-1.0, 0.0, 0.0]))  # candidates, in order of preference
VIEW_NAMES = ("front", "side", "top", "iso")
_FRAGMENTS = 4_000_000  # pixel tests per batch (memory bound)


def _axis(vector: np.ndarray) -> str:
    i = int(np.argmax(np.abs(vector)))
    return ("-" if vector[i] < 0 else "+") + "XYZ"[i]


def views(front: np.ndarray = DEFAULT_FRONT) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """view -> (direction the camera looks along, image right, label), for an object whose
    front faces ``front`` (a horizontal unit axis). With the default front (-Y): front looks
    +Y, side looks -X, top looks -Z with +X right, iso is from +X -Y +Z."""
    side = np.cross(_Z, front)  # the object's right-hand side, seen from the front
    iso = front + side + 0.8 * _Z
    return {
        "front": (-front, side, f"front (from {_axis(front)}; {_axis(side)} right, Z up)"),
        "side": (-side, -front, f"side (from {_axis(side)}; {_axis(-front)} right, Z up)"),
        "top": (-_Z, side, f"top (from +Z; {_axis(side)} right, {_axis(-front)} up)"),
        "iso": (-iso / np.linalg.norm(iso), side,
                f"iso (from {_axis(side)} {_axis(front)} +Z)"),
    }


VIEWS = views()  # the default facing, for the view names and the tool schema


def _rgb(color: int) -> np.ndarray:
    return np.array([(color >> 16) & 255, (color >> 8) & 255, color & 255], dtype=float)


def camera(forward: np.ndarray, right: np.ndarray | None = None):
    """(right, up, forward) unit vectors; ``right`` is a hint, made orthogonal."""
    if right is None:
        right = np.cross(forward, _Z)
        if np.linalg.norm(right) < 1e-9:  # looking straight down
            right = np.array([1.0, 0.0, 0.0])
    right = right - (right @ forward) * forward
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up / np.linalg.norm(up), forward


def _faces(meshes, fwd: np.ndarray, light: np.ndarray):
    """All visible-side triangles: (n, 3, 3) corners, (n, 3) colours, (n,) body index."""
    triangles, colors, owners = [], [], []
    for index, (_name, mesh) in enumerate(meshes):
        if len(mesh.faces) == 0:
            continue
        tri, normals = mesh.triangles, mesh.face_normals
        if mesh.is_watertight:  # closed, outward-facing (the mesh-health check enforces it)
            keep = normals @ fwd < 0  # back-face culling
            tri, normals = tri[keep], normals[keep]
            shade = 0.32 + 0.68 * np.clip(normals @ light, 0, 1)
        else:  # open surface: both sides can be seen
            shade = 0.32 + 0.68 * np.abs(normals @ light)
        base = _rgb(PALETTE[index % len(PALETTE)])
        triangles.append(tri)
        colors.append(np.clip(shade[:, None] * base[None, :], 0, 255))
        owners.append(np.full(len(tri), index))
    if not triangles:
        return np.zeros((0, 3, 3)), np.zeros((0, 3)), np.zeros(0, dtype=int)
    return np.concatenate(triangles), np.concatenate(colors), np.concatenate(owners)


def zbuffer(meshes, forward: np.ndarray, width: int, height: int,
            right: np.ndarray | None = None):
    """Rasterize with a depth test. Returns (face index per pixel, -1 = background; colours
    per face; body index per face; pixels per metre)."""
    right, up, fwd = camera(forward, right)
    light = -fwd + 0.6 * up - 0.4 * right  # upper left: the front is the iso view's left face
    tri, colors, owners = _faces(meshes, fwd, light / np.linalg.norm(light))
    face = np.full(width * height, -1, dtype=np.int64)
    if len(tri) == 0:
        return face.reshape(height, width), colors, owners, 1.0
    u, v, z = tri @ right, tri @ up, tri @ fwd  # (n, 3) each; larger z is farther
    margin = 0.08 * min(width, height)
    scale = min((width - 2 * margin) / max(u.max() - u.min(), 1e-9),
                (height - 2 * margin) / max(v.max() - v.min(), 1e-9))
    px = width / 2 + (u - (u.min() + u.max()) / 2) * scale
    py = height / 2 - (v - (v.min() + v.max()) / 2) * scale
    area = ((px[:, 1] - px[:, 0]) * (py[:, 2] - py[:, 0])
            - (px[:, 2] - px[:, 0]) * (py[:, 1] - py[:, 0]))
    # Scanline spans: each triangle gives one span of pixel centres per row it crosses, so the
    # work is the covered area, also for long thin triangles (extrusion and lathe caps).
    y0 = np.clip(np.ceil(py.min(axis=1) - 0.5), 0, height).astype(np.int64)
    y1 = np.clip(np.floor(py.max(axis=1) - 0.5), -1, height - 1).astype(np.int64)
    rows = np.maximum(y1 - y0 + 1, 0)
    rows[np.abs(area) <= 1e-12] = 0  # edge-on triangles cover no pixel
    span = np.clip(np.ceil(px.max(axis=1)) - np.floor(px.min(axis=1)) + 1, 1, width)
    work = np.cumsum(rows * (span / 2 + 1))  # about the fragments of each triangle
    depth = np.full(width * height, np.inf)
    ids = np.flatnonzero(rows > 0)
    cuts = np.searchsorted(work[ids], np.arange(1, int(work[-1] // _FRAGMENTS) + 2)
                           * _FRAGMENTS)
    for t in np.split(ids, np.unique(cuts)):
        if len(t) == 0:
            continue
        n_rows = rows[t]
        tri_r = np.repeat(t, n_rows)
        row = (np.arange(n_rows.sum()) - np.repeat(np.cumsum(n_rows) - n_rows, n_rows)
               + np.repeat(y0[t], n_rows))
        yc = row + 0.5
        low = np.full(len(tri_r), np.inf)
        high = np.full(len(tri_r), -np.inf)
        for k in range(3):  # where each edge crosses the row's centre line
            ax, ay = px[tri_r, k], py[tri_r, k]
            bx, by = px[tri_r, (k + 1) % 3], py[tri_r, (k + 1) % 3]
            crosses = (np.minimum(ay, by) <= yc) & (yc <= np.maximum(ay, by)) & (ay != by)
            with np.errstate(divide="ignore", invalid="ignore"):
                x = ax + (yc - ay) * (bx - ax) / (by - ay)
            low = np.where(crosses, np.minimum(low, x), low)
            high = np.where(crosses, np.maximum(high, x), high)
        xa = np.clip(np.ceil(low - 0.5), 0, width).astype(np.int64)
        xb = np.clip(np.floor(high - 0.5), -1, width - 1).astype(np.int64)
        count = np.where(np.isfinite(low), np.maximum(xb - xa + 1, 0), 0)
        who = np.repeat(tri_r, count)
        if len(who) == 0:
            continue
        Y = np.repeat(row, count)
        X = (np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
             + np.repeat(xa, count))
        cx, cy = X + 0.5, Y + 0.5
        ax, ay, bx, by = px[who, 0], py[who, 0], px[who, 1], py[who, 1]
        qx, qy = px[who, 2], py[who, 2]
        w0 = ((bx - cx) * (qy - cy) - (qx - cx) * (by - cy)) / area[who]
        w1 = ((qx - cx) * (ay - cy) - (ax - cx) * (qy - cy)) / area[who]
        frag_z = w0 * z[who, 0] + w1 * z[who, 1] + (1.0 - w0 - w1) * z[who, 2]
        pixel = Y * width + X
        order = np.lexsort((frag_z, pixel))  # by pixel, nearest first
        pixel, frag_z, who = pixel[order], frag_z[order], who[order]
        first = np.r_[True, pixel[1:] != pixel[:-1]]
        pixel, frag_z, who = pixel[first], frag_z[first], who[first]
        nearer = frag_z < depth[pixel]
        depth[pixel[nearer]] = frag_z[nearer]
        face[pixel[nearer]] = who[nearer]
    return face.reshape(height, width), colors, owners, scale


def rasterize(meshes, forward: np.ndarray, label: str,
              right: np.ndarray | None = None) -> Image.Image:
    """meshes: [(name, world trimesh)] -> an RGB image of one view."""
    width, height = WIDTH * SUPERSAMPLE, HEIGHT * SUPERSAMPLE
    face, colors, _owners, _scale = zbuffer(meshes, forward, width, height, right)
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:] = BACKGROUND
    hit = face >= 0
    pixels[hit] = colors[face[hit]].astype(np.uint8)
    image = Image.fromarray(pixels, "RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), label, fill=(31, 41, 51))
    for index, (name, _mesh) in enumerate(meshes):
        y = HEIGHT - 16 * (len(meshes) - index) - 4
        draw.rectangle((8, y + 2, 18, y + 12), fill=tuple(int(c) for c in
                                                         _rgb(PALETTE[index % len(PALETTE)])))
        draw.text((24, y + 1), name, fill=(31, 41, 51))
    return image


# ---- facing -----------------------------------------------------------------------------------


def moving_bodies(build: Build) -> set[str]:
    """Bodies that move when the driven joints move (mimic and loop followers included)."""
    rest = solve_pose(build, {}).transforms
    for fraction in (0.5, 0.25, 0.05):
        pose = {}
        for driver in build.loaded.drivers:
            low, high = driver["lower"], driver["upper"]
            pose[driver["name"]] = fraction * (high if abs(high) >= abs(low) else low)
        try:
            moved = build.loaded.assembly.solve(pose).transforms
        except (LoopClosureError, PoseError):
            continue
        return {name for name, matrix in moved.items()
                if not np.allclose(matrix, rest[name], atol=1e-7)}
    root = build.loaded.manifest["articulation"]["root"]
    return {body.name for body in build.loaded.assembly.bodies if body.name != root}


def front_of(meshes, moving: set[str]) -> np.ndarray:
    """The horizontal side whose front view (plus a quarter of its iso view) shows the largest
    visible area of moving bodies, in square metres, so framing does not matter; -Y unless
    another side shows clearly more (10% more). The straight view decides: from an iso angle,
    two neighbouring sides both see a door's face edge-on."""
    if not moving:
        return DEFAULT_FRONT
    names = [name for name, _mesh in meshes]
    is_moving = np.array([name in moving for name in names])

    def share(forward, right) -> float:
        face, _colors, owners, scale = zbuffer(meshes, forward, 160, 120, right)
        hit = face[face >= 0]
        return float(is_moving[owners[hit]].sum()) / scale ** 2  # m2 of moving bodies seen

    scores = []
    for front in FRONTS:
        directions = views(front)
        scores.append(share(*directions["front"][:2]) + 0.25 * share(*directions["iso"][:2]))
    best = int(np.argmax(scores))
    if scores[best] > 1.1 * scores[0]:
        return FRONTS[best]
    return DEFAULT_FRONT


def build_front(build: Build) -> np.ndarray:
    """The build's front at rest, computed once per build."""
    if build.front is None:
        rest = world_meshes(build, solve_pose(build, {}).transforms)
        build.front = front_of(rest, moving_bodies(build))
    return build.front


# ---- the tool ---------------------------------------------------------------------------------


def _next_folder(ctx: ToolContext):
    root = ctx.run_dir / "renders"
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    folder = root / f"{max(numbers, default=0) + 1:04d}"
    folder.mkdir()
    return folder


def save_png(image: Image.Image, path: Path) -> None:
    """Save a PNG atomically (a crash never leaves half an image)."""
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    atomic_write_bytes(path, buffer.getvalue())


def run_render(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    names = args.get("views") or list(VIEW_NAMES)
    if len(set(names)) != len(names):
        raise ToolError("bad_arguments", f"views has duplicates: {names}", "list each view once")
    build = newest_build(ctx)
    pose = solve_pose(build, args.get("pose") or {})
    meshes = world_meshes(build, pose.transforms)
    directions = views(build_front(build))
    folder = _next_folder(ctx)
    images = []
    for view in names:
        forward, right, label = directions[view]
        path = folder / f"{view}.png"
        save_png(rasterize(meshes, forward, f"{build.label}  {label}", right), path)
        images.append({"view": view, "path": path.relative_to(ctx.run_dir).as_posix(),
                       "label": label})
    bodies = {}
    for name, mesh in meshes:
        low, high = mesh.bounds
        bodies[name] = {"min_m": round_list(low), "max_m": round_list(high),
                        "size_mm": round_list((high - low) * 1000, 1)}
    every = np.vstack([m.bounds for _n, m in meshes])
    result = {"build": build.label, "images": images,
              "joints": joint_report(build, pose.joint_values), "bodies": bodies,
              "overall_size_mm": round_list((every.max(axis=0) - every.min(axis=0)) * 1000, 1)}
    if build.note:
        result["note"] = build.note
    result["_images"] = [str(folder / f"{view}.png") for view in names]
    return result


RENDER = AgentTool(
    "render",
    "Render the newest build at a pose to PNG views (front, side, top, iso; facing the "
    "object's front) with each body's bounding box. Use to check shape and proportions.",
    {"type": "object", "properties": {
        "pose": POSE_SCHEMA,
        "views": {"type": "array", "items": {"type": "string", "enum": list(VIEW_NAMES)}},
    }, "additionalProperties": False},
    run_render)
