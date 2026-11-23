"""The agent's tools, called directly (no LLM)."""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest
from PIL import Image

import llm
from agent.tools import BY_NAME, TOOLS, AgentTool, ToolContext, run_tool, schema_size, schemas
from agent.tools.core import cap_text, to_text

sys.path.insert(0, os.path.dirname(__file__))
from check_fixtures import DRAWER_SOURCE  # noqa: E402

FOUR_BAR = "from examples.four_bar import build\n\nobject_model = build()\n"
LAPTOP = "from examples.laptop import build\n\nobject_model = build()\n"
STUCK_FOUR_BAR = ("from examples.four_bar import build\n\n"
                  "object_model = build(rocker_limits=(-0.05, 0.05))\n")


def call(ctx, name, args=None):
    out = run_tool(ctx, name, args if args is not None else {})
    assert json.loads(out.text) == out.data  # the model sees exactly this JSON
    return out


def ok(ctx, name, args=None) -> dict:
    out = call(ctx, name, args)
    assert not out.is_error, out.text
    return out.data


def error(ctx, name, args=None) -> dict:
    out = call(ctx, name, args)
    assert out.is_error, out.text
    assert set(out.data) == {"error_kind", "message", "hint"}  # never a bare string
    return out.data


@pytest.fixture()
def ctx(tmp_path):
    return ToolContext(tmp_path / "run")


def compiled(tmp_path_factory, source: str) -> ToolContext:
    context = ToolContext(tmp_path_factory.mktemp("run"))
    (context.workspace / "main.py").write_text(source, encoding="utf-8")
    run_tool(context, "compile", {})
    return context


@pytest.fixture(scope="module")
def four_bar(tmp_path_factory):
    return compiled(tmp_path_factory, FOUR_BAR)


@pytest.fixture(scope="module")
def drawer(tmp_path_factory):
    return compiled(tmp_path_factory, DRAWER_SOURCE)


# ---- registry and schemas -----------------------------------------------------------------------


def test_registry_order_schemas_and_size() -> None:
    names = [t.name for t in TOOLS]
    assert names == ["read", "write", "edit", "compile", "render", "probe"]
    assert [s.name for s in schemas()] == names and schemas() == schemas()  # fixed order
    assert all(isinstance(s, llm.Tool) for s in schemas())
    assert not any(n in BY_NAME for n in ("exec_command", "shell", "bash", "run"))  # no shell
    for tool in TOOLS:
        assert "\n" not in tool.description and len(tool.description) < 170, tool.name
        assert tool.parameters["type"] == "object"
    assert [t.name for t in TOOLS if t.parallel_ok] == ["read"]
    size = schema_size()
    print(f"\ntool schemas: {size['chars']} chars, about {size['tokens_estimate']} tokens")
    assert size["tokens_estimate"] < 1000


def test_bad_arguments_are_structured(ctx) -> None:
    assert error(ctx, "read", {})["message"] == "arguments: missing required 'path'"
    assert "must be integer" in error(ctx, "read", {"path": "a", "offset": "3"})["message"]
    assert "unknown field 'mode'" in error(ctx, "write", {"path": "a", "content": "",
                                                          "mode": "x"})["message"]
    assert error(ctx, "shell", {"command": "dir"})["error_kind"] == "unknown_tool"
    assert error(ctx, "read", ["main.py"])["error_kind"] == "bad_arguments"


def test_a_tool_bug_is_still_a_structured_error(ctx, monkeypatch) -> None:
    def broken(_ctx, _args):
        raise KeyError("oops")

    monkeypatch.setitem(BY_NAME, "boom", AgentTool("boom", "Breaks.", {"type": "object"}, broken))
    data = error(ctx, "boom")
    assert data["error_kind"] == "internal_error" and "KeyError" in data["message"]


def test_output_is_capped_with_head_and_tail() -> None:
    text = "".join(f"line {i}\n" for i in range(5000))
    capped = cap_text(text, 2000)
    assert len(capped) <= 2000 and capped.startswith("line 0\n") and capped.endswith("line 4999\n")
    assert "characters of text omitted (head and tail shown)" in capped
    data = {"path": "main.py", "text": text}
    out = to_text(data, 4000)
    assert len(out) <= 4000 and json.loads(out)["path"] == "main.py"


def test_tool_output_becomes_a_tool_result(ctx) -> None:
    message = call(ctx, "read", {"path": "."}).message("call_7")
    assert message == llm.ToolResult("call_7", "read", message.content, False)


# ---- paths --------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["../secret.txt", "a/../../x", "docs/sdk/../../README.md",
                                  "/etc/passwd", "C:\\Windows\\win.ini", "C:main.py",
                                  "\\\\server\\share\\x"])
def test_paths_cannot_escape(ctx, path) -> None:
    for tool, args in (("read", {"path": path}), ("write", {"path": path, "content": "x"}),
                       ("edit", {"path": path, "edits": [{"old_text": "a", "new_text": "b"}]})):
        assert error(ctx, tool, args)["error_kind"] == "path_refused", (tool, path)


def test_symlink_out_of_the_workspace_is_refused(ctx, tmp_path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (ctx.workspace / "link.txt").symlink_to(outside)
    except OSError:
        pytest.skip("cannot create symlinks here (Windows without developer mode)")
    assert error(ctx, "read", {"path": "link.txt"})["error_kind"] == "path_refused"


def test_docs_are_readable_but_read_only(ctx) -> None:
    listing = ok(ctx, "read", {"path": "docs/sdk"})
    assert listing["path"] == "docs/sdk" and "README.md" in listing["entries"]
    assert ok(ctx, "read", {"path": "docs/sdk/README.md"})["text"].startswith("L1: # SDK docs")
    data = error(ctx, "write", {"path": "docs/sdk/notes.md", "content": "x"})
    assert data["error_kind"] == "path_refused" and "read-only" in data["message"]
    assert error(ctx, "edit", {"path": "docs/sdk/README.md", "edits": [
        {"old_text": "SDK", "new_text": "X"}]})["error_kind"] == "path_refused"


# ---- read / write / edit ------------------------------------------------------------------------


def test_write_then_read_with_line_numbers(ctx) -> None:
    content = "".join(f"value_{i} = {i}\n" for i in range(1, 11))
    assert ok(ctx, "write", {"path": "pkg/model.py", "content": content}) == {
        "path": "pkg/model.py", "created": True, "bytes": len(content), "lines": 10}
    assert ok(ctx, "write", {"path": "pkg/model.py", "content": content})["created"] is False
    data = ok(ctx, "read", {"path": "pkg/model.py", "offset": 3, "limit": 2})
    assert data["text"] == "L3: value_3 = 3\nL4: value_4 = 4"
    assert data["lines"] == "3-4" and data["total_lines"] == 10 and data["next_offset"] == 5
    assert "next_offset" not in ok(ctx, "read", {"path": "pkg/model.py"})
    assert ok(ctx, "read", {"path": "."})["entries"] == ["pkg/"]


def test_read_errors(ctx) -> None:
    assert error(ctx, "read", {"path": "missing.py"})["error_kind"] == "not_found"
    (ctx.workspace / "image.bin").write_bytes(bytes(range(256)))
    assert error(ctx, "read", {"path": "image.bin"})["error_kind"] == "not_text"
    (ctx.workspace / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert "past the end" in error(ctx, "read", {"path": "a.py", "offset": 9})["message"]


def test_big_reads_are_capped(ctx) -> None:
    (ctx.workspace / "big.py").write_text("".join(f"# {'x' * 70} {i}\n" for i in range(2000)),
                                          encoding="utf-8")
    out = call(ctx, "read", {"path": "big.py", "limit": 2000})
    assert len(out.text) <= 16_000 and "omitted (head and tail shown)" in out.data["text"]
    assert out.data["text"].startswith("L1: ") and out.data["text"].rstrip().endswith(" 1999")


def test_edit_applies_all_edits(ctx) -> None:
    (ctx.workspace / "main.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    data = ok(ctx, "edit", {"path": "main.py", "edits": [
        {"old_text": "a = 1", "new_text": "a = 10"}, {"old_text": "c = 3\n", "new_text": ""}]})
    assert data["replaced"] == 2
    assert (ctx.workspace / "main.py").read_text(encoding="utf-8") == "a = 10\nb = 2\n"


def test_edit_no_match_multi_match_and_all_or_nothing(ctx) -> None:
    original = "def f():\n    return 1\n\nx = f()\ny = f()\n"
    path = ctx.workspace / "main.py"
    path.write_text(original, encoding="utf-8")
    data = error(ctx, "edit", {"path": "main.py", "edits": [
        {"old_text": "x = f()", "new_text": "x = 2"},
        {"old_text": "def f():\n  return 1", "new_text": "def f():\n    return 2"}]})
    assert data["error_kind"] == "edit_failed"
    assert data["message"].startswith("edit 2 of 2: old_text was not found")
    assert "matches at line 1 if whitespace is ignored" in data["hint"]
    assert path.read_text(encoding="utf-8") == original  # edit 1 was not applied either
    data = error(ctx, "edit", {"path": "main.py", "edits": [
        {"old_text": "f()\n", "new_text": "g()\n"}]})
    assert "matches 2 times (lines 4, 5)" in data["message"]
    assert "more surrounding lines" in data["hint"]
    assert "old_text is empty" in error(ctx, "edit", {"path": "main.py", "edits": [
        {"old_text": "", "new_text": "z"}]})["message"]
    assert error(ctx, "edit", {"path": "nope.py", "edits": [
        {"old_text": "a", "new_text": "b"}]})["error_kind"] == "not_found"
    assert path.read_text(encoding="utf-8") == original


# ---- compile ------------------------------------------------------------------------------------


def test_compile_a_broken_main_returns_the_signal_block(ctx) -> None:
    (ctx.workspace / "main.py").write_text("from sdk import Assembly\nobject_model = Assembly(\n",
                                           encoding="utf-8")
    data = ok(ctx, "compile")  # the tool worked; the compile failed
    assert data["status"] == "error" and data["version"] is None and data["cached"] is False
    block = data["compile_signals"]
    assert block.startswith("<compile_signals>") and block.rstrip().endswith("</compile_signals>")
    assert "syntax" in block.lower() and "main.py" in block
    assert set(data) == {"status", "version", "cached", "compile_signals"}
    assert ok(ctx, "compile")["cached"] is True  # nothing changed


def test_compile_success(four_bar) -> None:
    data = ok(four_bar, "compile")
    assert data["status"] == "success" and data["version"] == "v0000"
    assert "Compile passed" in data["compile_signals"]


# ---- render -------------------------------------------------------------------------------------


def test_render_writes_four_pngs_for_the_laptop(tmp_path_factory) -> None:
    context = compiled(tmp_path_factory, LAPTOP)
    out = call(context, "render", {"pose": {"hinge": 1.2}})
    assert not out.is_error, out.text
    data = out.data
    assert [i["view"] for i in data["images"]] == ["front", "side", "top", "iso"]
    assert len(out.images) == 4
    for image_path, entry in zip(out.images, data["images"], strict=True):
        assert image_path.is_file() and image_path.suffix == ".png"
        assert (context.run_dir / entry["path"]) == image_path
        with Image.open(image_path) as image:
            assert image.size == (640, 480)
            assert len(image.getcolors(maxcolors=100_000) or []) > 50  # shaded, not blank
    assert set(data["bodies"]) == {"base", "lid"}
    assert data["joints"]["hinge"]["deg"] == pytest.approx(math.degrees(1.2), abs=1e-3)
    for box in data["bodies"].values():
        assert all(hi >= lo for lo, hi in zip(box["min_m"], box["max_m"], strict=True))
    print(f"\nrender: {out.images[3]}")
    again = ok(context, "render", {"views": ["top"]})
    assert [i["view"] for i in again["images"]] == ["top"]
    assert again["images"][0]["path"] == "renders/0002/top.png"
    assert error(context, "render", {"views": ["back"]})["error_kind"] == "bad_arguments"


def test_render_and_probe_need_a_build(ctx) -> None:
    for name, args in (("render", {}), ("probe", {"queries": [{"type": "collisions"}]})):
        data = error(ctx, name, args)
        assert data["error_kind"] == "no_build" and "compile" in data["hint"]


# ---- probe --------------------------------------------------------------------------------------


def test_probe_four_bar_rocker_angle_matches_solve(four_bar) -> None:
    from examples.four_bar import build

    crank = math.radians(60)
    expected = build().solve({"crank_joint": crank})
    data = ok(four_bar, "probe", {"pose": {"crank_joint": crank}, "queries": [
        {"type": "position", "body": "rocker"},
        {"type": "point", "body": "coupler", "point": [0.05, 0, 0]},
        {"type": "collisions"}]})
    rocker = expected.joint_values["rocker_joint"]
    assert data["joints"]["rocker_joint"]["rad"] == pytest.approx(rocker, abs=1e-5)
    assert data["joints"]["rocker_joint"]["deg"] == pytest.approx(math.degrees(rocker), abs=1e-3)
    assert data["joints"]["crank_joint"]["deg"] == pytest.approx(60.0)
    position, point, collisions = data["results"]
    assert position["origin_m"] == pytest.approx(expected.transforms["rocker"][:3, 3], abs=1e-6)
    world = expected.transforms["coupler"] @ np.array([0.05, 0, 0, 1])
    assert point["world_m"] == pytest.approx(world[:3], abs=1e-6)
    assert collisions["pairs"] == []  # as the checks say: a clean linkage
    print(f"\nprobe: {json.dumps(data)}")


def test_probe_reports_the_drawer_collision_at_270_mm(drawer) -> None:
    data = ok(drawer, "probe", {"pose": {"slide": 0.27}, "queries": [
        {"type": "collisions"}, {"type": "gap", "a": "cabinet", "b": "drawer"}]})
    assert data["build"] == "last_failed" and "failed its checks" in data["note"]
    collisions, gap = data["results"]
    (pair,) = collisions["pairs"]
    assert (pair["a"], pair["b"]) == ("cabinet", "drawer")
    assert pair["over_tolerance"] and pair["depth_mm"] > collisions["tolerance_mm"]
    assert gap["state"] == "penetrating" and gap["gap_mm"] == 0.0
    assert gap["depth_mm"] == pytest.approx(pair["depth_mm"])
    clear = ok(drawer, "probe", {"pose": {"slide": 0.1}, "queries": [{"type": "collisions"}]})
    assert clear["results"][0]["pairs"] == []


def test_probe_bad_input(four_bar, tmp_path_factory) -> None:
    def kind(args):
        return error(four_bar, "probe", args)

    assert "no body named 'lid'" in kind({"queries": [{"type": "position",
                                                       "body": "lid"}]})["message"]
    assert "needs 'b'" in kind({"queries": [{"type": "gap", "a": "crank"}]})["message"]
    assert "point must be [x, y, z]" in kind({"queries": [
        {"type": "point", "body": "crank", "point": [1, 2]}]})["message"]
    data = kind({"pose": {"rocker_joint": 0.1}, "queries": [{"type": "collisions"}]})
    assert "moved by a mimic or loop" in data["message"] and "crank_joint" in data["hint"]
    assert "not a joint" in kind({"pose": {"nope": 1}, "queries": [
        {"type": "collisions"}]})["message"]
    assert kind({"queries": []})["error_kind"] == "bad_arguments"
    stuck = compiled(tmp_path_factory, STUCK_FOUR_BAR)  # fails its loop-reach check
    data = error(stuck, "probe", {"pose": {"crank_joint": math.pi},
                                  "queries": [{"type": "collisions"}]})
    assert data["error_kind"] == "pose_unreachable" and "pin" in data["message"]
    assert not call(stuck, "probe", {"pose": {"crank_joint": 0.02},
                                     "queries": [{"type": "collisions"}]}).is_error
