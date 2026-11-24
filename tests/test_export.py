"""Export: URDF, GLB, meshes, manifest and versions, checked on all 8 examples."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from sdk import Assembly, Box, BoxGeometry, ValidationError
from sdk.export import COLLISION_MAX_FACES, GLB_ROOT, export, load_manifest
from sdk.transforms import invert, rpy_matrix
from urdf_fk import read_urdf

# Example -> joint values to compare at (besides the rest pose).
POSES = {
    "laptop": [{"hinge": 0.7}, {"hinge": 2.3}],
    "four_bar": [{"crank_joint": 1.0}, {"crank_joint": 4.0}],
    "gantry": [{"x_drive": 0.3, "y_head": -0.2}, {"x_drive": 0.85, "y_head": 0.3}],
    "toaster": [{"lever_slide": -0.08, "browning": 2.0}],
    "door_hinge": [{"pin": 0.4}, {"pin": 2.5}],
    "desk_lamp": [{"swivel": 1.0, "tilt": -0.3}, {"swivel": -2.5, "tilt": 0.05}],
    "computer_mouse": [{"left_button_click": -0.03, "right_button_click": -0.01, "scroll": 4.0}],
    "folding_chair": [{"fold": -0.5}],
}


def module(name: str):
    return importlib.import_module(f"examples.{name}")


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> dict[str, tuple[Assembly, object]]:
    """Every example built and exported once, shared by the tests in this file."""
    root = tmp_path_factory.mktemp("exports")
    out = {}
    for name in POSES:
        model = module(name).build()
        out[name] = (model, export(model, root / name, source=module(name).__file__))
    return out


# ---- URDF ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(POSES))
def test_urdf_is_valid_and_complete(exported, name: str) -> None:
    model, result = exported[name]
    art = model.validate()
    urdf = read_urdf(result.urdf)  # parses as XML
    assert urdf.name == model.name
    assert set(urdf.links) == {b.name for b in model.bodies}
    assert set(urdf.joints) == {j.name for j in art.tree}  # closures are not in the URDF
    for link in urdf.links.values():
        for mesh in link.iter("mesh"):
            assert (result.directory / mesh.get("filename")).is_file()
        assert float(link.find("inertial/mass").get("value")) > 0
    for joint in art.tree:
        exported_joint = urdf.joints[joint.name]
        if joint.mimic is None:
            assert exported_joint.mimic is None
        else:
            assert exported_joint.mimic == (joint.mimic.source, joint.mimic.multiplier,
                                            joint.mimic.offset)
        if joint.type.value == "revolute" and joint.limits is None:
            assert exported_joint.type == "continuous"
        elif joint.limits is not None:
            assert exported_joint.limits == pytest.approx(joint.limits)


def test_gantry_has_mimic_tags(exported) -> None:
    _, result = exported["gantry"]
    text = result.urdf.read_text(encoding="utf-8")
    assert '<mimic joint="x_drive" multiplier="1" offset="0" />' in text
    assert '<mimic joint="x_drive" multiplier="0.5" offset="0" />' in text
    assert "bridge_right" not in text  # the loop closure stays out of the tree


@pytest.mark.parametrize("name", list(POSES))
def test_urdf_forward_kinematics_matches_solve(exported, name: str) -> None:
    model, result = exported[name]
    urdf = read_urdf(result.urdf)
    for values in [{}, *POSES[name]]:
        pose = model.solve(values)
        # Feed the URDF every tree joint value the solver used (driven, rest, or solved
        # around a loop); the URDF computes mimic joints itself.
        tree_values = {j: v for j, v in pose.joint_values.items()
                       if j in urdf.joints and urdf.joints[j].mimic is None}
        from_urdf = urdf.body_poses(tree_values)
        for body, transform in pose.transforms.items():
            assert from_urdf[body] == pytest.approx(transform, abs=1e-9), (name, values, body)


# ---- GLB -------------------------------------------------------------------------------------


def rest_bounds_y_up(model: Assembly) -> np.ndarray:
    pose = model.solve({}, warm_start={})
    meshes = []
    for body in model.bodies:
        mesh = body.trimesh()
        mesh.apply_transform(pose.transforms[body.name])
        meshes.append(mesh)
    points = trimesh.util.concatenate(meshes).vertices
    y_up = np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])
    return np.array([y_up.min(axis=0), y_up.max(axis=0)])


@pytest.mark.parametrize("name", list(POSES))
def test_glb_loads_with_one_node_per_body(exported, name: str) -> None:
    model, result = exported[name]
    scene = trimesh.load(result.glb)
    assert isinstance(scene, trimesh.Scene)
    nodes = set(scene.graph.nodes) - {scene.graph.base_frame}
    assert nodes == {b.name for b in model.bodies} | {GLB_ROOT}
    parents = {child: parent for parent, child, *_ in scene.graph.to_edgelist()}
    art = model.validate()
    assert parents[art.root] == GLB_ROOT
    for joint in art.tree:  # the node tree is the joint tree
        assert parents[joint.child.body.name] == joint.parent.body.name
    assert scene.bounds == pytest.approx(rest_bounds_y_up(model), abs=1e-5)


# ---- manifest --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(POSES))
def test_manifest_round_trips_and_matches(exported, name: str) -> None:
    model, result = exported[name]
    manifest = load_manifest(result.directory)
    assert json.loads(json.dumps(manifest)) == manifest
    art = model.validate()
    assert manifest["articulation"]["closures"] == [j.name for j in art.closures]
    assert [c["name"] for c in manifest["closures"]] == [j.name for j in art.closures]
    assert manifest["articulation"]["root"] == art.root
    assert [m["joint"] for m in manifest["mimics"]] == [j.name for j in art.mimic_order]
    assert {b["name"] for b in manifest["bodies"]} == {b.name for b in model.bodies}
    data = (result.directory / manifest["source"]["file"]).read_bytes()
    assert manifest["source"]["sha256"] == hashlib.sha256(data).hexdigest()

    # The rest pose, rebuilt from the manifest's joint frames alone (as a viewer would),
    # matches the stored rest pose and the assembly.
    stored = {k: np.array(v) for k, v in manifest["rest_pose"]["body_transforms"].items()}
    rest = model.solve({}, warm_start={})
    joints = {j["name"]: j for j in manifest["joints"]}
    rebuilt = {art.root: np.eye(4)}
    for joint_name in manifest["articulation"]["tree"]:
        j = joints[joint_name]
        frame = lambda f: _frame(f["xyz"], f["rpy"])  # noqa: E731
        q = manifest["rest_pose"]["joint_values"].get(joint_name, 0.0)
        rebuilt[j["child"]] = (rebuilt[j["parent"]] @ frame(j["parent_frame"])
                               @ _motion(j, q) @ invert(frame(j["child_frame"])))
    for body in rest.transforms:
        assert stored[body] == pytest.approx(rest.transforms[body], abs=1e-12)
        assert rebuilt[body] == pytest.approx(rest.transforms[body], abs=1e-12)


def _frame(xyz, rpy) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rpy_matrix(rpy)
    t[:3, 3] = xyz
    return t


def _motion(joint: dict, q: float) -> np.ndarray:
    t = np.eye(4)
    if joint["type"] == "revolute":
        k = np.asarray(joint["axis"])
        skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        t[:3, :3] = np.eye(3) + math.sin(q) * skew + (1 - math.cos(q)) * skew @ skew
    elif joint["type"] == "prismatic":
        t[:3, 3] = np.asarray(joint["axis"]) * q
    return t


def test_four_bar_closure_is_in_the_manifest(exported) -> None:
    _, result = exported["four_bar"]
    manifest = load_manifest(result.manifest)
    (pin,) = manifest["closures"]
    assert pin["name"] == "pin" and pin["role"] == "closure" and pin["urdf_type"] is None
    assert pin["parent"] == "coupler" and pin["child"] == "rocker"


# ---- meshes and mass -------------------------------------------------------------------------


def test_big_bodies_get_simplified_collision_meshes(exported) -> None:
    _, result = exported["toaster"]
    shell = next(b for b in load_manifest(result.manifest)["bodies"] if b["name"] == "body")
    assert shell["triangles"] > COLLISION_MAX_FACES
    assert shell["collision_triangles"] <= COLLISION_MAX_FACES
    assert shell["collision_method"] == "manifold_simplify"
    collision = trimesh.load(result.directory / shell["collision_mesh"])
    visual = trimesh.load(result.directory / shell["visual_mesh"])
    assert collision.bounds == pytest.approx(visual.bounds, abs=2e-3)


def test_mass_and_inertia_of_a_box(tmp_path: Path) -> None:
    model = Assembly("brick")
    brick = model.body("brick")
    brick.add("block", BoxGeometry((0.2, 0.1, 0.05)), xyz=(0.5, 0, 0))
    result = export(model, tmp_path, density=2000.0)
    (body,) = load_manifest(result.manifest)["bodies"]
    mass = 0.2 * 0.1 * 0.05 * 2000.0
    assert body["mass"] == pytest.approx(mass)
    assert body["center_of_mass"] == pytest.approx((0.5, 0, 0))
    expected = np.diag([mass / 12 * (0.1**2 + 0.05**2), mass / 12 * (0.2**2 + 0.05**2),
                        mass / 12 * (0.2**2 + 0.1**2)])
    assert np.array(body["inertia"]) == pytest.approx(expected, abs=1e-12)
    heavier = export(model, tmp_path, densities={"brick": 8000.0})
    assert load_manifest(heavier.manifest)["bodies"][0]["mass"] == pytest.approx(mass * 4)


# ---- versions and refusal -------------------------------------------------------------------


def tree_hash(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(folder).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_versions_increment_and_never_overwrite(tmp_path: Path) -> None:
    model = module("laptop").build()
    first = export(model, tmp_path)
    before = tree_hash(first.directory)
    second = export(model, tmp_path)
    third = export(model, tmp_path)
    assert [first.version, second.version, third.version] == [0, 1, 2]
    assert [p.name for p in sorted(tmp_path.iterdir())] == ["v0000", "v0001", "v0002"]
    with pytest.raises(FileExistsError):
        export(model, tmp_path, version=1)
    assert export(model, tmp_path, version=7).directory.name == "v0007"
    assert export(model, tmp_path).version == 8
    assert tree_hash(first.directory) == before  # untouched by later exports
    assert load_manifest(second.directory)["version"] == 1


def test_invalid_assembly_is_refused(tmp_path: Path) -> None:
    model = Assembly("broken")
    model.body("base").add("box", Box((0.1, 0.1, 0.1)))
    model.body("floating")  # no shapes and no joint
    with pytest.raises(ValidationError):
        export(model, tmp_path)
    assert list(tmp_path.iterdir()) == []  # nothing written, not even a temp folder


def test_missing_source_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        export(module("laptop").build(), tmp_path, source=tmp_path / "nope.py")
    assert list(tmp_path.iterdir()) == []


# ---- step 10 regressions ----------------------------------------------------------------------


def test_simplifying_a_thin_shell_keeps_it_solid() -> None:
    """A 3 mm sheet-metal tunnel: doubling the simplify tolerance used to collapse its walls
    to nothing ("simplify produced an empty result")."""
    from sdk import ExtrudeWithHolesGeometry
    from sdk.export import _simplify

    def ring(r, n=200):
        return [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
                for i in range(n)]

    shell = ExtrudeWithHolesGeometry(ring(0.09), [ring(0.087)], 0.46).to_trimesh(process=True)
    assert shell.is_watertight
    reduced, method = _simplify(shell, budget=100)
    assert len(reduced.faces) > 0
    assert abs(reduced.volume - shell.volume) <= 0.1 * shell.volume
    assert method in {"manifold_simplify", "copy"}


def test_loading_a_mesh_with_an_open_piece_needs_no_repair() -> None:
    """viewer.loader splits body meshes into pieces; an open piece must not trigger trimesh's
    hole filling (it needs networkx, which is not a dependency, and changes the geometry)."""
    from viewer.loader import _pieces

    closed = trimesh.creation.box((1, 1, 1))
    open_piece = trimesh.Trimesh(vertices=[[3, 0, 0], [4, 0, 0], [3, 1, 0], [4, 1, 1]],
                                 faces=[[0, 1, 2], [1, 3, 2]])
    pieces = _pieces(trimesh.util.concatenate([closed, open_piece]))
    assert len(pieces) == 2
    assert sorted(len(p.faces) for _n, p in pieces) == [2, 12]


def test_open_fragments_load_as_one_shape() -> None:
    """Zero-area triangles can split off as one-triangle pieces. They must not each become a
    shape: the collision scene is pairwise (step 10: 35,095 shapes, a 13-minute probe)."""
    from viewer.loader import _pieces

    boxes = [trimesh.creation.box((1, 1, 1)).apply_translation((3 * i, 0, 0)) for i in range(2)]
    fragments = [trimesh.Trimesh(vertices=[[i, 5, 0], [i + 0.5, 5, 0], [i + 1, 5, 0]],
                                 faces=[[0, 1, 2]], process=False) for i in range(3000)]
    pieces = dict(_pieces(trimesh.util.concatenate(boxes + fragments)))
    assert sorted(pieces) == ["open", "piece1", "piece2"]
    assert len(pieces["open"].faces) == 3000
    assert all(pieces[name].is_watertight for name in ("piece1", "piece2"))
