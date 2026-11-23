"""The viewer: loader, HTTP server, and the page's JavaScript kinematics (run under Node)."""

from __future__ import annotations

import importlib
import json
import math
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from sdk.export import export
from viewer.loader import load_version
from viewer.server import STATIC, ViewerServer

EXAMPLES = ["laptop", "four_bar", "gantry", "toaster", "door_hinge", "desk_lamp",
            "computer_mouse", "folding_chair"]
Z_UP_TO_Y_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


def build(name: str, **kwargs):
    return importlib.import_module(f"examples.{name}").build(**kwargs)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("viewer")
    return {name: export(build(name), root / name).directory for name in EXAMPLES}


# ---- loader -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXAMPLES)
def test_loader_poses_match_the_original_assembly(exported, name: str) -> None:
    original = build(name)
    loaded = load_version(exported[name])
    assert {b.name for b in loaded.assembly.bodies} == {b.name for b in original.bodies}
    assert [j.name for j in loaded.assembly.joints] == [j.name for j in original.joints]
    for fraction in (0.0, 0.3, 0.85):
        values = {d["name"]: d["lower"] + (d["upper"] - d["lower"]) * fraction
                  for d in loaded.drivers}
        expected = original.solve(values).transforms
        actual = loaded.assembly.solve(values).transforms
        for body, matrix in expected.items():
            assert actual[body] == pytest.approx(matrix, abs=1e-9), (name, fraction, body)


def test_loader_modes_and_sliders(exported) -> None:
    four_bar, gantry = load_version(exported["four_bar"]), load_version(exported["gantry"])
    assert four_bar.mode == "loop" and [d["name"] for d in four_bar.drivers] == ["crank_joint"]
    assert four_bar.drivers[0]["lower"] == pytest.approx(-math.pi)  # unlimited: a full turn
    # The gantry has a closure, but its mimics leave nothing to solve: the browser poses it.
    assert gantry.mode == "tree"
    assert [d["name"] for d in gantry.drivers] == ["x_drive", "y_head"]


# ---- server -----------------------------------------------------------------------------------


def get(url: str):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


def post(url: str, body: dict):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.fixture()
def four_bar_server(tmp_path):
    export(build("four_bar"), tmp_path / "result")
    export(build("four_bar", rocker_limits=(-0.05, 0.05)), tmp_path / "result")  # v0001
    server = ViewerServer(tmp_path).start()
    yield server
    server.stop()


def test_server_serves_page_vendored_three_and_versions(four_bar_server) -> None:
    url = four_bar_server.url
    status, kind, page = get(url)
    assert status == 200 and kind.startswith("text/html") and b"importmap" in page
    status, kind, three = get(url + "static/vendor/three/three.module.js")
    assert status == 200 and kind.startswith("text/javascript")  # not text/plain on Windows
    assert b"REVISION = '170'" in three
    assert (STATIC / "vendor" / "three" / "LICENSE").read_text().startswith("The MIT License")
    status, _, body = get(url + "api/versions")
    assert [v["id"] for v in json.loads(body)["versions"]] == ["v0001", "v0000"]
    status, _, body = get(url + "api/version/v0000")
    described = json.loads(body)
    assert described["mode"] == "loop" and described["manifest"]["name"] == "four_bar"
    status, kind, glb = get(url + "api/version/v0000/model.glb")
    assert glb[:4] == b"glTF" and kind == "model/gltf-binary"
    with pytest.raises(urllib.error.HTTPError):
        get(url + "static/../loader.py")  # no path traversal out of static/


def test_pose_endpoint_solves_the_four_bar(four_bar_server) -> None:
    expected = build("four_bar").solve({"crank_joint": 1.0}).transforms
    status, body = post(four_bar_server.url + "api/version/v0000/pose", {"crank_joint": 1.0})
    assert status == 200 and body["ok"]
    for name, matrix in expected.items():
        assert np.array(body["bodies"][name]).reshape(4, 4) == pytest.approx(matrix, abs=1e-9)
    assert "rocker_joint" in body["joints"]  # the solved loop values come back too


def test_pose_endpoint_rejects_unreachable_poses_cleanly(four_bar_server) -> None:
    url = four_bar_server.url + "api/version/v0001/pose"  # rocker limited to +-0.05 rad
    assert post(url, {"crank_joint": 0.02})[0] == 200
    status, body = post(url, {"crank_joint": math.pi})
    assert status == 409 and not body["ok"]
    assert "loop closure 'pin' stays open" in body["error"]
    status, body = post(url, {"crank_joint": 0.01})  # the model is still usable afterwards
    assert status == 200 and body["ok"]
    assert post(url, {"crank_joint": "not a number"})[0] == 400


# ---- the page's JavaScript, under Node ----------------------------------------------------------

NODE = shutil.which("node")


def run_page_js(tmp_path: Path, version: Path, poses: list[dict]) -> dict:
    """Stage the page's modules and the vendored three.js, then run the harness with Node.

    Node has no import map, so ``node_modules/three`` stands in for it: ``three`` and
    ``three/addons/...`` resolve to the vendored files, as in the browser."""
    stage = tmp_path / "js"
    vendor = STATIC / "vendor" / "three"
    package = stage / "node_modules" / "three"
    shutil.copytree(vendor / "addons", package / "addons")
    shutil.copy(vendor / "three.module.js", package / "three.module.js")
    (package / "package.json").write_text(json.dumps(
        {"name": "three", "type": "module",
         "exports": {".": "./three.module.js", "./addons/*": "./addons/*"}}))
    (stage / "package.json").write_text('{"type": "module"}')
    for module in ("kinematics.js", "display.js"):
        shutil.copy(STATIC / module, stage / module)
    shutil.copy(Path(__file__).parent / "js" / "pose_glb.mjs", stage / "pose_glb.mjs")
    completed = subprocess.run(
        [NODE, str(stage / "pose_glb.mjs"), str(version / "model.glb"),
         str(version / "model.json"), json.dumps(poses)],
        capture_output=True, text=True, timeout=60, check=False)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
@pytest.mark.parametrize("name", ["laptop", "gantry"])
def test_page_kinematics_match_python(exported, tmp_path: Path, name: str) -> None:
    loaded = load_version(exported[name])
    poses = [{}]
    for fraction in (0.25, 0.9):
        poses.append({d["name"]: d["lower"] + (d["upper"] - d["lower"]) * fraction
                      for d in loaded.drivers})
    results = run_page_js(tmp_path, exported[name], poses)["poses"]
    original = build(name)
    for pose, result in zip(poses, results, strict=True):
        expected = original.solve(pose).transforms
        for body, matrix in expected.items():
            fk = np.array(result["fk"][body]).reshape(4, 4)
            assert fk == pytest.approx(matrix, abs=1e-9), (name, pose, body)  # JS = Python
            node = np.array(result["nodes"][body]).reshape(4, 4)  # GLB node, Y-up, float32
            assert node == pytest.approx(Z_UP_TO_Y_UP @ matrix, abs=1e-5), (name, pose, body)


def hue_saturation(rgb: int) -> tuple[float, float]:
    import colorsys

    r, g, b = ((rgb >> 16) & 255) / 255, ((rgb >> 8) & 255) / 255, (rgb & 255) / 255
    h, _l, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360, s


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
@pytest.mark.parametrize("name", ["four_bar", "desk_lamp"])
def test_bodies_are_lit_coloured_and_outlined(exported, tmp_path: Path, name: str) -> None:
    """The exported GLB has no normals (lit materials then render black): the page adds them,
    gives each body its own palette colour, an edge outline, and red only when highlighted."""
    result = run_page_js(tmp_path, exported[name], [{}])
    style, palette, red = result["style"], result["palette"], result["highlight"]
    for color in palette:  # nothing red or pink (hue within 25 deg of red) in the palette
        hue, saturation = hue_saturation(color)
        assert saturation < 0.15 or 25 <= hue <= 320, f"#{color:06x}"
    assert len(set(palette)) == len(palette) and red not in palette
    colors = [entry["color"] for entry in style.values()]
    assert colors == palette[:len(colors)]  # by body index, so distinct up to 10 bodies
    for body, entry in style.items():
        assert not entry["hadNormals"], body  # if export starts writing normals, they are kept
        assert entry["lit"] and entry["normals"] >= entry["vertices"] > 0, body
        assert entry["worstNormalError"] < 1e-4, body
        assert entry["edgeSegments"] > 0, body
        assert entry["highlighted"] == red and entry["restored"] == entry["color"], body


# ---- failed compiles reach the viewer ----------------------------------------------------------


def test_failed_check_compile_is_viewable(tmp_path: Path) -> None:
    from check_fixtures import DRAWER_SOURCE
    from compiler import compile

    run_dir = tmp_path / "drawer"
    (run_dir / "workspace").mkdir(parents=True)
    (run_dir / "workspace" / "main.py").write_text(DRAWER_SOURCE)
    result = compile(run_dir, timeout=120)
    assert result.status == "error"
    assert not [p for p in (run_dir / "result").iterdir() if p.name.startswith("v")]
    server = ViewerServer(run_dir).start()
    try:
        versions = json.loads(get(server.url + "api/versions")[2])["versions"]
        assert [v["id"] for v in versions] == ["last_failed"] and versions[0]["failed"]
        described = json.loads(get(server.url + "api/version/last_failed")[2])
        (signal,) = [s for s in described["signals"] if s["severity"] == "failure"]
        assert signal["kind"] == "swept_collision"
        assert signal["bodies"] == ["cabinet", "drawer"]
        assert signal["pose"]["slide"] == pytest.approx(0.26, abs=0.002)
    finally:
        server.stop()


def test_cli_rejects_a_missing_folder(tmp_path: Path) -> None:
    from viewer.__main__ import main

    assert main([str(tmp_path / "nope"), "--no-browser"]) == 2
