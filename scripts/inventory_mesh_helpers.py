"""Inventory Articraft's sdk/_mesh public helpers (AST only, no imports of the target)."""
import ast, json, re, sys
from pathlib import Path

repo = Path(sys.argv[1])
sdk = repo / "src/articraft/sdk"
mesh_dir = sdk / "_mesh"
EXTERNAL = {"np": "numpy", "numpy": "numpy", "trimesh": "trimesh", "manifold3d": "manifold3d",
            "igl": "libigl", "scipy": "scipy", "build123d": "build123d", "math": None}

# Public = exported by sdk/__init__ or sdk/mesh.py and defined in _mesh.
def exported(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and "_mesh" in node.module:
            names |= {a.name for a in node.names}
    return names
public = exported(sdk / "__init__.py") | exported(sdk / "mesh.py")

defs = {}  # name -> (file, node)
module_imports = {}
for f in sorted(mesh_dir.glob("*.py")):
    tree = ast.parse(f.read_text(encoding="utf-8"))
    imps = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names: imps[a.asname or a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names: imps[a.asname or a.name] = node.module
    module_imports[f.name] = imps
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defs[node.name] = (f, node)

def names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
           {n.value.id for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}

def closure(name, seen=None):
    seen = set() if seen is None else seen
    if name in seen or name not in defs: return seen
    seen.add(name)
    for ref in names_in(defs[name][1]):
        if ref in defs and ref.startswith("_"):
            closure(ref, seen)
    return seen

def deps(name):
    f, node = defs[name]
    out = set()
    for member in closure(name):
        mf, mnode = defs[member]
        imps = module_imports[mf.name]
        for ref in names_in(mnode):
            mod = imps.get(ref)
            if mod is None: continue
            top = mod.split(".")[0]
            if top in ("numpy", "trimesh", "manifold3d", "igl", "scipy", "build123d", "PIL"):
                out.add({"igl": "libigl"}.get(top, top))
    return sorted(out)

def internal_calls(name):
    _, node = defs[name]
    return sorted(r for r in names_in(node) if r in defs and not r.startswith("_") and r != name)

# Usage corpora
corpora = {
    "examples": [*(repo / "examples").rglob("*.py"), *(sdk / "docs/examples").glob("*.py")],
    "quickstart": [sdk / "docs/common/00_quickstart.md"],
    "docs": [p for p in [*(sdk / "docs/common").glob("*.md"), *(sdk / "docs/mesh").glob("*.md")]
             if p.name != "00_quickstart.md"],
    "system_prompt": [repo / "src/articraft/prompts/system.md"],
}
texts = {k: [p.read_text(encoding="utf-8") for p in v] for k, v in corpora.items()}

rows = []
for name in sorted(public):
    if name not in defs: continue
    f, node = defs[name]
    own = node.end_lineno - node.lineno + 1
    total = sum(defs[m][1].end_lineno - defs[m][1].lineno + 1 for m in closure(name))
    doc = (ast.get_docstring(node) or "").strip().splitlines()
    pat = re.compile(rf"\b{re.escape(name)}\b")
    usage = {k: sum(len(pat.findall(t)) for t in ts) for k, ts in texts.items()}
    usage_files = {k: sum(1 for t in ts if pat.search(t)) for k, ts in texts.items()}
    sig = ""
    if isinstance(node, ast.FunctionDef):
        sig = ast.unparse(node.args)
    else:
        init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init: sig = ast.unparse(init.args).replace("self, ", "", 1)
        else:  # dataclass fields
            sig = ", ".join(ast.unparse(n.target) + (": " + ast.unparse(n.annotation) if n.annotation else "")
                            for n in node.body if isinstance(n, ast.AnnAssign))
    rows.append(dict(name=name, kind="class" if isinstance(node, ast.ClassDef) else "func",
                     file=f.name, own=own, total=total, doc=doc[0] if doc else "", deps=deps(name),
                     calls=internal_calls(name), usage=usage, usage_files=usage_files, sig=sig))
json.dump(rows, open(sys.argv[2], "w"), indent=1)
print(len(rows), "public helpers;", len(public - set(defs)), "exported names not in _mesh:", sorted(public - set(defs))[:60])
