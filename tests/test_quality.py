"""Step 10: quality metrics on known examples, the final renders, prompt sets in the loop,
planning effort, and the new record fields. No network."""

from __future__ import annotations

import json
import re

import pytest
from PIL import Image

from agent.loop import PROMPT_SETS, EffortPolicy, best_build, run
from agent.quality import helpers_used, quality, render_final
from compiler import compile as compile_run
from llm import Budget, ScriptedModel

QUICKSTART = PROMPT_SETS["v1"].quickstart
MAILBOX = re.findall(r"```python\n(.*?)```", QUICKSTART.read_text(encoding="utf-8"), re.S)[-1]
LAPTOP = re.findall(r"```python\n(.*?)```",
                    PROMPT_SETS["v0"].quickstart.read_text(encoding="utf-8"), re.S)[-1]
BAD_MAILBOX = MAILBOX.replace("limits=(0.0, math.radians(100))", "limits=(0.5, 1.0)")
assert BAD_MAILBOX != MAILBOX


def built(tmp_path_factory, name: str, source: str):
    run_dir = tmp_path_factory.mktemp(name)
    (run_dir / "workspace").mkdir()
    (run_dir / "workspace" / "main.py").write_text(source, encoding="utf-8")
    assert compile_run(run_dir, timeout=180).status == "success"
    return run_dir


@pytest.fixture(scope="module")
def mailbox(tmp_path_factory):
    return built(tmp_path_factory, "mailbox", MAILBOX)


@pytest.fixture(scope="module")
def laptop(tmp_path_factory):
    return built(tmp_path_factory, "laptop", LAPTOP)


# ---- metrics --------------------------------------------------------------------------------


def test_quality_of_the_mailbox(mailbox) -> None:
    q = quality(mailbox / "result" / "v0000")
    assert (q["bodies"], q["shapes"], q["joints"], q["movable_joints"]) == (3, 8, 2, 2)
    assert q["non_box_share"] == 0.75  # arm and flag plate are boxes
    assert q["helpers"] == ["ExtrudeGeometry", "ExtrudeWithHolesGeometry", "LatheGeometry",
                            "RoundedBoxGeometry"]
    assert q["helper_count"] == 4
    assert q["shape_types"] == {"Box": 2, "Cylinder": 1, "ExtrudeGeometry": 2,
                                "ExtrudeWithHolesGeometry": 1, "LatheGeometry": 1,
                                "RoundedBoxGeometry": 1}
    model = json.loads((mailbox / "result" / "v0000" / "model.json").read_text(encoding="utf-8"))
    assert q["triangles"] == sum(b["triangles"] for b in model["bodies"]) > 1000


def test_quality_of_the_all_box_laptop(laptop) -> None:
    q = quality(laptop / "result" / "v0000")
    assert (q["bodies"], q["shapes"], q["joints"]) == (2, 4, 1)
    assert q["non_box_share"] == 0.0
    assert q["helpers"] == [] and q["helper_count"] == 0
    assert q["triangles"] == 4 * 12


def test_helpers_count_uses_not_imports() -> None:
    source = ("import sdk\nfrom sdk import Box, LatheGeometry, rounded_rect_profile, "
              "ExtrudeGeometry\nshape = ExtrudeGeometry.centered(rounded_rect_profile(1, 1, 0.1), "
              "0.1)\nknob = sdk.TorusGeometry(1, 0.1)\nbox = Box((1, 1, 1))\n")
    assert helpers_used(source) == ["ExtrudeGeometry", "TorusGeometry", "rounded_rect_profile"]
    assert helpers_used("def broken(:\n") == []


def test_render_final_writes_four_views_and_a_sheet(mailbox) -> None:
    paths = render_final(mailbox, best_build(mailbox))
    assert paths == [f"renders/final/{v}.png" for v in ("front", "side", "top", "iso", "views")]
    sizes = [Image.open(mailbox / p).size for p in paths]
    assert sizes[:4] == [(640, 480)] * 4 and sizes[4] == (1280, 960)
    assert render_final(mailbox, None) == []


# ---- the loop -------------------------------------------------------------------------------


def write(content: str) -> dict:
    return {"tool_calls": [{"name": "write", "arguments": {"path": "main.py",
                                                           "content": content}}],
            "usage": {"output_tokens": 500, "reasoning_tokens": 400}}


COMPILE = {"tool_calls": [{"name": "compile", "arguments": {}}],
           "usage": {"output_tokens": 50, "reasoning_tokens": 30}}
FIX = {"tool_calls": [{"name": "edit", "arguments": {"path": "main.py", "edits": [
    {"old_text": "limits=(0.5, 1.0)", "new_text": "limits=(0.0, math.radians(100))"}]}}],
    "usage": {"output_tokens": 200, "reasoning_tokens": 150}}
DONE = "Done: a mailbox with a flap (0-100 deg) and a flag (0-90 deg)."


def go(tmp_path, replies, **kwargs):
    budget = Budget(max_cost_usd=1.0, max_calls=20)
    model = ScriptedModel(replies, budget=budget)
    result = run("a mailbox with a hinged flap", model, budget, run_dir=tmp_path / "run",
                 **kwargs)
    return result, model


def test_default_prompt_set_is_v1_and_v0_is_step9s(tmp_path) -> None:
    result, model = go(tmp_path / "a", [write(MAILBOX), COMPILE, DONE])
    messages = model.requests[0][0]
    assert messages[0].content == PROMPT_SETS["v1"].system.read_text(encoding="utf-8")
    assert "mailbox" in messages[1].content and "short plan" in messages[2].content
    assert result.record["prompt_set"] == "v1"

    result, model = go(tmp_path / "b", [write(LAPTOP), COMPILE, DONE], prompt_set="v0")
    messages = model.requests[0][0]
    assert messages[0].content == PROMPT_SETS["v0"].system.read_text(encoding="utf-8")
    assert "laptop" in messages[1].content and "Plan the bodies" in messages[2].content
    assert result.record["prompt_set"] == "v0"


def test_each_prompt_set_sees_its_own_docs_folder(tmp_path) -> None:
    read = {"tool_calls": [{"name": "read", "arguments": {"path": "docs/sdk/shapes.md"}}]}
    for name, found in (("v1", True), ("v0", False)):
        result, model = go(tmp_path / name, [read, write(MAILBOX), COMPILE, DONE],
                           prompt_set=name)
        reply = model.requests[1][0][-1].content
        assert ("# Shapes" in reply) is found, reply[:200]
        assert result.record["doc_reads"] == {"docs/sdk/shapes.md": 1}


def test_planning_effort_applies_until_the_first_compile(tmp_path) -> None:
    replies = [write(BAD_MAILBOX), COMPILE, FIX, COMPILE, DONE]
    result, model = go(tmp_path, replies, effort=EffortPolicy(planning_effort="low"))
    assert result.status == "success"
    assert model.options == [{"effort": "low"}, {"effort": "low"},  # planning
                             {}, {},                                  # fixing (compile failed)
                             {"effort": "low", "max_output_tokens": 16_000}]  # routine
    assert result.record["effort_turns"] == {"planning": 2, "default": 2, "routine": 1}


def test_no_planning_effort_by_default(tmp_path) -> None:
    result, model = go(tmp_path, [write(MAILBOX), COMPILE, DONE], effort=EffortPolicy())
    assert model.options[:2] == [{}, {}]
    assert "planning" not in result.record["effort_turns"]


def test_record_has_planning_tokens_quality_and_final_renders(tmp_path) -> None:
    replies = [write(BAD_MAILBOX), COMPILE, FIX, COMPILE, DONE]
    result, _ = go(tmp_path, replies)
    record = json.loads((result.run_dir / "record.json").read_text(encoding="utf-8"))
    assert record["first_compile_turn"] == 2
    assert record["reasoning_before_first_compile"] == 400 + 30
    assert record["output_before_first_compile"] == 500 + 50
    assert record["quality"]["bodies"] == 3 and record["quality"]["helper_count"] == 4
    assert record["final_renders"][-1] == "renders/final/views.png"
    assert all((result.run_dir / p).is_file() for p in record["final_renders"])
    assert record["reminders"] == {}


def test_a_run_without_a_build_has_no_quality(tmp_path) -> None:
    result, _ = go(tmp_path, ["I will think about it.", "Still thinking.", "Hmm."])
    assert result.status == "no_progress"
    assert result.record["quality"] is None and result.record["final_renders"] == []
    assert result.record["first_compile_turn"] is None
    assert result.record["reminders"] == {"compile_required": 2}
