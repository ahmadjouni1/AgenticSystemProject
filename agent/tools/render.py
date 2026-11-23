"""render: the newest build at a pose, as PNG views, plus a text summary.

CPU only (numpy + Pillow): orthographic views, flat Lambert shading, back-face culling on
closed bodies, painter's algorithm (triangles drawn far to near), 2x supersampling. Colours match the viewer's palette, which
has no reds (red is kept for collisions).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from agent.tools.builds import (POSE_SCHEMA, joint_report, newest_build, round_list,
                                solve_pose, world_meshes)
from agent.tools.core import AgentTool, ToolContext, ToolError

WIDTH, HEIGHT, SUPERSAMPLE = 640, 480, 2
BACKGROUND = (243, 244, 246)
PALETTE = [0x4E79A7, 0xF28E2B, 0x76B7B2, 0x59A14F, 0xEDC948, 0xB07AA1, 0xB59B5A, 0xA0CBE8,
           0x8CD17D, 0xBAB0AC]  # = viewer/static/display.js
_Z = np.array([0.0, 0.0, 1.0])
# view -> (direction the camera looks along, label). World is Z-up.
VIEWS = {
    "front": (np.array([0.0, 1.0, 0.0]), "front (looking +Y; X right, Z up)"),
    "side": (np.array([-1.0, 0.0, 0.0]), "side (looking -X; Y right, Z up)"),
    "top": (np.array([0.0, 0.0, -1.0]), "top (looking -Z; X right, Y up)"),
    "iso": (np.array([-1.0, 1.0, -0.8]) / np.linalg.norm([1.0, 1.0, 0.8]),
            "iso (from +X -Y +Z)"),
}


def _rgb(color: int) -> np.ndarray:
    return np.array([(color >> 16) & 255, (color >> 8) & 255, color & 255], dtype=float)


def camera(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    right = np.cross(forward, _Z)
    if np.linalg.norm(right) < 1e-9:  # looking straight down
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up / np.linalg.norm(up), forward


def rasterize(meshes, forward: np.ndarray, label: str) -> Image.Image:
    """meshes: [(name, world trimesh)] -> an RGB image of one view."""
    right, up, fwd = camera(forward)
    scale_up = SUPERSAMPLE
    width, height = WIDTH * scale_up, HEIGHT * scale_up
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    triangles, colors, depths = [], [], []
    light = -fwd + 0.6 * up + 0.4 * right
    light /= np.linalg.norm(light)
    for index, (_name, mesh) in enumerate(meshes):
        if len(mesh.faces) == 0:
            continue
        tri, normals = mesh.triangles, mesh.face_normals  # (n, 3, 3), (n, 3)
        if mesh.is_watertight:  # closed, outward-facing (the mesh-health check enforces it)
            front = normals @ fwd < 0  # back-face culling: hidden faces are not drawn
            tri, normals = tri[front], normals[front]
            shade = 0.32 + 0.68 * np.clip(normals @ light, 0, 1)
        else:  # open surface: both sides can be seen
            shade = 0.32 + 0.68 * np.abs(normals @ light)
        base = _rgb(PALETTE[index % len(PALETTE)])
        triangles.append(tri)
        colors.append(np.clip(shade[:, None] * base[None, :], 0, 255))
        depths.append(tri.mean(axis=1) @ fwd)
    if triangles:
        tri = np.concatenate(triangles)
        color = np.concatenate(colors).astype(int)
        depth = np.concatenate(depths)
        u, v = tri @ right, tri @ up  # (n, 3) each
        low_u, high_u, low_v, high_v = u.min(), u.max(), v.min(), v.max()
        margin = 0.08 * min(width, height)
        scale = min((width - 2 * margin) / max(high_u - low_u, 1e-9),
                    (height - 2 * margin) / max(high_v - low_v, 1e-9))
        cx, cy = (low_u + high_u) / 2, (low_v + high_v) / 2
        px = width / 2 + (u - cx) * scale
        py = height / 2 - (v - cy) * scale
        for i in np.argsort(-depth):  # far first
            fill = tuple(color[i])
            draw.polygon([(px[i, 0], py[i, 0]), (px[i, 1], py[i, 1]), (px[i, 2], py[i, 2])],
                         fill=fill, outline=fill)
    image = image.resize((WIDTH, HEIGHT), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), label, fill=(31, 41, 51))
    for index, (name, _mesh) in enumerate(meshes):
        y = HEIGHT - 16 * (len(meshes) - index) - 4
        draw.rectangle((8, y + 2, 18, y + 12), fill=tuple(int(c) for c in
                                                         _rgb(PALETTE[index % len(PALETTE)])))
        draw.text((24, y + 1), name, fill=(31, 41, 51))
    return image


def _next_folder(ctx: ToolContext):
    root = ctx.run_dir / "renders"
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    folder = root / f"{max(numbers, default=0) + 1:04d}"
    folder.mkdir()
    return folder


def run_render(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    views = args.get("views") or list(VIEWS)
    if len(set(views)) != len(views):
        raise ToolError("bad_arguments", f"views has duplicates: {views}", "list each view once")
    build = newest_build(ctx)
    pose = solve_pose(build, args.get("pose") or {})
    meshes = world_meshes(build, pose.transforms)
    folder = _next_folder(ctx)
    images = []
    for view in views:
        forward, label = VIEWS[view]
        path = folder / f"{view}.png"
        rasterize(meshes, forward, f"{build.label}  {label}").save(path)
        images.append({"view": view, "path": path.relative_to(ctx.run_dir).as_posix()})
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
    result["_images"] = [str(folder / f"{view}.png") for view in views]
    return result


RENDER = AgentTool(
    "render",
    "Render the newest build at a pose to PNG views (front, side, top, iso) with each body's "
    "bounding box. Use to check shape and proportions.",
    {"type": "object", "properties": {
        "pose": POSE_SCHEMA,
        "views": {"type": "array", "items": {"type": "string", "enum": list(VIEWS)}},
    }, "additionalProperties": False},
    run_render)
