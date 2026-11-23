// Test harness, run by tests/test_viewer.py under Node, in a temp folder where
// node_modules/three is the vendored three.js (so the add-ons' bare "three" import resolves,
// as the page's import map does in the browser).
//
//   node pose_glb.mjs <model.glb> <model.json> '<[{joint: value}, ...]>'
//
// For each pose it prints:
// - fk: kinematics.js body transforms (row-major, Z-up), exactly what the page computes;
// - nodes: the GLB's body nodes after the page's posing step (localMatrices into
//   node.matrix), read back as world matrices from three.js (row-major, Y-up).
// It also prints `style`: what display.js did to each body (normals, colour, edges), and
// the palette.

import { readFileSync } from "node:fs";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { HIGHLIGHT, PALETTE, setHighlighted, styleBody } from "./display.js";
import { bodyTransforms, localMatrices } from "./kinematics.js";

const [glbPath, manifestPath, posesJson] = process.argv.slice(2);
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const poses = JSON.parse(posesJson);
const data = readFileSync(glbPath);
const buffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);

new GLTFLoader().parse(buffer, "", (gltf) => {
  const names = manifest.bodies.map((b) => b.name);
  const hadNormals = {};
  const style = {};
  names.forEach((name, i) => {
    const node = gltf.scene.getObjectByName(name);
    hadNormals[name] = Boolean(node.geometry.getAttribute("normal"));
    const material = styleBody(node, i, names);
    const normal = node.geometry.getAttribute("normal");
    let worst = 0;
    for (let k = 0; k < normal.count; k++) {
      const length = Math.hypot(normal.getX(k), normal.getY(k), normal.getZ(k));
      worst = Math.max(worst, Math.abs(length - 1));
    }
    const color = material.color.getHex();
    setHighlighted(material, true, i);
    const highlighted = material.color.getHex();
    setHighlighted(material, false, i);
    const edges = node.children.filter((c) => c.isLineSegments);
    style[name] = {
      hadNormals: hadNormals[name], normals: normal.count, vertices: node.geometry.attributes.position.count,
      worstNormalError: worst, color, highlighted, restored: material.color.getHex(),
      lit: material.isMeshStandardMaterial,
      edgeSegments: edges.length ? edges[0].geometry.attributes.position.count / 2 : 0,
    };
  });
  const results = poses.map((pose) => {
    const world = bodyTransforms(manifest, pose);
    const local = localMatrices(manifest, world);
    const nodes = {};
    for (const [name, matrix] of Object.entries(local)) {
      const node = gltf.scene.getObjectByName(name);
      node.matrixAutoUpdate = false;
      node.matrix.set(...matrix);
    }
    gltf.scene.updateMatrixWorld(true);
    for (const body of manifest.bodies) {
      const e = gltf.scene.getObjectByName(body.name).matrixWorld.elements; // column-major
      nodes[body.name] = [0, 1, 2, 3].flatMap((r) => [0, 1, 2, 3].map((c) => e[c * 4 + r]));
    }
    return { fk: world, nodes };
  });
  console.log(JSON.stringify({ poses: results, style, palette: PALETTE, highlight: HIGHLIGHT }));
}, (error) => {
  console.error(error.message);
  process.exit(1);
});
