"""The system prompt and the SDK docs (step 10): sizes, honesty of every example, and that
no docs example builds an evaluation object.

- Every ```python block in docs/sdk/*.md is run. A block that defines ``object_model`` is a
  complete program: it goes through the real compiler and must pass every check. Any other
  block must run, and every mesh it makes must be closed.
- ```text blocks hold signatures, not programs: every function or method they (or any inline
  code span in the docs or the system prompt) call must exist in the SDK.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
from pathlib import Path

import pytest

import sdk
from agent.loop import PROMPT_SETS, REPLICA_ROOT
from compiler import compile as compile_run
from sdk.testing import TestContext

DOCS = REPLICA_ROOT / "docs" / "sdk"
PAGES = sorted(DOCS.glob("*.md"))
SYSTEM = REPLICA_ROOT / "prompts" / "system.md"
EVAL_PROMPTS = json.loads((REPLICA_ROOT / "prompts" / "eval_prompts.json")
                          .read_text(encoding="utf-8"))
MAX_TOKENS = 2000
# Docs examples that build an object some evaluation prompt also asks for, and why.
ALLOWED_EVAL_OBJECTS = {
    "four_bar": "step 10 asked for four-bar authoring with frame_in in joints_and_loops.md; "
                "the four-bar was an evaluation prompt in steps 8-9 only",
}
TOOLS = {"read", "write", "edit", "compile", "render", "probe"}


CHARS_PER_TOKEN = 2.9


def tokens(text: str) -> int:
    """A conservative token count. DeepSeek counted 2.96-3.95 characters per token on these
    files (scripts/doc_tokens.py, 2026-11-23); code-heavy pages are near 3, so the usual
    4-characters estimate would let a page grow to about 2.6k real tokens."""
    return -(-len(text) * 10 // int(CHARS_PER_TOKEN * 10))


def blocks(page: Path, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", page.read_text(encoding="utf-8"), re.S)


PYTHON = [(page.name, i, code) for page in PAGES for i, code in enumerate(blocks(page, "python"))]
PROGRAMS = [(name, i, code) for name, i, code in PYTHON if "object_model" in code]
SNIPPETS = [(name, i, code) for name, i, code in PYTHON if "object_model" not in code]


def object_name(code: str) -> str:
    """The name passed to Assembly(...) in a program."""
    for node in ast.walk(ast.parse(code)):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Assembly"
                and node.args and isinstance(node.args[0], ast.Constant)):
            return node.args[0].value
    raise AssertionError("no Assembly(...) in the program")


# ---- sizes and structure --------------------------------------------------------------------


def test_the_system_prompt_is_at_most_2k_tokens() -> None:
    assert tokens(SYSTEM.read_text(encoding="utf-8")) <= MAX_TOKENS


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_each_docs_page_is_at_most_2k_tokens(page: Path) -> None:
    assert tokens(page.read_text(encoding="utf-8")) <= MAX_TOKENS


def test_the_system_prompt_lists_every_docs_page_and_says_the_quickstart_is_in_context() -> None:
    text = SYSTEM.read_text(encoding="utf-8")
    on_demand = {p.name for p in PAGES} - {"README.md", "quickstart.md"}
    assert on_demand == {"shapes.md", "joints_and_loops.md", "testing.md"}
    for name in on_demand:
        assert re.search(rf"^- `{re.escape(name)}`: .+", text, re.M), name  # one line each
    assert "already in your context" in text and "never read" in text


def test_the_docs_use_only_python_and_text_blocks() -> None:
    for page in PAGES:
        fences = re.findall(r"^```(\w*)$", page.read_text(encoding="utf-8"), re.M)
        assert set(fences[0::2]) <= {"python", "text"}, page.name  # every opening fence


def test_the_quickstart_is_the_second_system_message_of_the_default_prompt_set() -> None:
    assert PROMPT_SETS["v1"].quickstart == DOCS / "quickstart.md"
    assert PROMPT_SETS["v1"].docs_dir == DOCS


# ---- every example is real ------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled(tmp_path_factory) -> dict[str, Path]:
    """Each complete docs program, compiled once: {"page#i": run folder}."""
    out = {}
    for name, i, code in PROGRAMS:
        run_dir = tmp_path_factory.mktemp(f"{Path(name).stem}{i}")
        (run_dir / "workspace").mkdir()
        (run_dir / "workspace" / "main.py").write_text(code, encoding="utf-8")
        out[f"{name}#{i}"] = run_dir
        out[f"{name}#{i}:result"] = compile_run(run_dir, timeout=180)
    return out


def test_there_are_complete_programs_in_the_quickstart_joints_and_testing_pages() -> None:
    assert {name for name, _i, _c in PROGRAMS} == {"quickstart.md", "joints_and_loops.md",
                                                   "testing.md"}


@pytest.mark.parametrize("name,i,code", PROGRAMS, ids=[f"{n}#{i}" for n, i, _ in PROGRAMS])
def test_every_docs_program_compiles_and_passes_its_checks(compiled, name, i, code) -> None:
    result = compiled[f"{name}#{i}:result"]
    assert result.status == "success", result.signals_text
    assert "warnings=0" in result.signals_text, result.signals_text


@pytest.mark.parametrize("name,i,code", SNIPPETS, ids=[f"{n}#{i}" for n, i, _ in SNIPPETS])
def test_every_docs_snippet_runs_and_makes_closed_meshes(name, i, code) -> None:
    namespace: dict = {}
    exec(compile(code, f"{name}#{i}", "exec"), namespace)  # noqa: S102 - our own docs
    meshes = {k: v for k, v in namespace.items() if isinstance(v, sdk.MeshGeometry)}
    assert meshes, "a snippet should build something"
    for key, mesh in meshes.items():
        assert mesh.to_trimesh(process=True).is_watertight, key


def _calls(code: str) -> list[tuple[bool, str]]:
    """(is a method, name) for every ``name(`` in a code span or signature block."""
    return [(dot == ".", name) for dot, name in re.findall(r"(\.?)\b([A-Za-z_]\w*)\(", code)]


def test_every_call_named_in_the_docs_and_prompt_exists() -> None:
    owners = (sdk.Assembly, sdk.Body, sdk.MeshGeometry, sdk.ExtrudeGeometry,
              sdk.LatheGeometry, TestContext)
    texts = {page.name: page.read_text(encoding="utf-8") for page in PAGES}
    texts["system.md"] = SYSTEM.read_text(encoding="utf-8")
    missing = []
    for name, text in texts.items():
        spans = re.findall(r"```text\n(.*?)```", text, re.S)
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        spans += re.findall(r"`([^`]+)`", prose)
        for span in spans:
            for is_method, call in _calls(span):
                ok = (any(hasattr(o, call) for o in owners) if is_method
                      else call in sdk.__all__ or call in TOOLS or call == "run_tests"
                      or hasattr(builtins, call))
                if not ok:
                    missing.append(f"{name}: {call}( in {span[:60]!r}")
    assert not missing, "\n".join(missing)


# ---- no docs example is an evaluation object -----------------------------------------------


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower().replace("_", " ").replace("-", " ")))


def matches_eval_prompt(name: str) -> list[str]:
    """Evaluation prompts that ask for the object called ``name`` (any word of 4+ letters)."""
    words = {w for w in _words(name) if len(w) >= 4}
    prompts = [p for step, ps in EVAL_PROMPTS.items() if not step.startswith("_") for p in ps]
    return [p for p in prompts if words & _words(p)]


def test_the_quickstart_example_is_not_any_evaluation_prompt() -> None:
    (code,) = blocks(DOCS / "quickstart.md", "python")
    name = object_name(code)
    assert name == "mailbox"
    assert matches_eval_prompt(name) == []
    assert "laptop" not in (DOCS / "quickstart.md").read_text(encoding="utf-8").lower()


def test_no_docs_program_builds_an_evaluation_object_unless_allowed() -> None:
    for page, _i, code in PROGRAMS:
        name = object_name(code)
        if name in ALLOWED_EVAL_OBJECTS:
            assert not set(matches_eval_prompt(name)) & set(EVAL_PROMPTS["step10"])
            continue
        assert matches_eval_prompt(name) == [], (page, name)


def test_the_matcher_catches_an_evaluation_object() -> None:
    assert matches_eval_prompt("folding_table") == ["a folding table"]
    assert matches_eval_prompt("laptop") == ["a laptop that opens and closes"]


def test_the_frozen_step9_prompt_set_is_unchanged() -> None:
    v0 = PROMPT_SETS["v0"]
    assert v0.system.read_text(encoding="utf-8").startswith("You are a mechanical designer")
    assert "## Quality rules" in v0.system.read_text(encoding="utf-8")
    assert 'Assembly("laptop")' in v0.quickstart.read_text(encoding="utf-8")
    assert sorted(p.name for p in v0.docs_dir.iterdir()) == ["README.md", "quickstart.md"]
