// How bodies look: a fixed colour per body index, lit materials, normals when the GLB has
// none, and thin dark edge outlines. Red is reserved for the collision highlight.

import * as THREE from "three";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";

// No reds or pinks, so a highlighted body is never confused with a normal one.
export const PALETTE = [
  0x4e79a7, // steel blue
  0xf28e2b, // orange
  0x76b7b2, // teal
  0x59a14f, // green
  0xedc948, // gold
  0xb07aa1, // purple
  0xb59b5a, // khaki
  0xa0cbe8, // light blue
  0x8cd17d, // light green
  0xbab0ac, // grey
];
export const HIGHLIGHT = 0xe5383b;
export const HIGHLIGHT_EMISSIVE = 0x4a0000;
const EDGE_COLOR = 0x1f2933;
const CREASE = THREE.MathUtils.degToRad(30); // sharper than this stays a hard edge
export const EDGE_NAME = "__edges__";

export function bodyColor(index) {
  return PALETTE[index % PALETTE.length];
}

// The exported GLB has positions only. Without normals a lit material renders black, so
// compute them: smooth across shallow angles (cylinders look round), split at creases
// (boxes keep flat faces).
export function withNormals(geometry) {
  if (geometry.getAttribute("normal")) return geometry;
  return toCreasedNormals(geometry, CREASE);
}

// A body's own meshes, not the child bodies nested under its node (or our edge lines).
export function ownMeshes(node, bodyNames) {
  if (node.isMesh) return [node];
  return node.children.filter((c) => c.isMesh && !bodyNames.includes(c.name));
}

// Give the body at `index` its colour, normals and edge outlines; return its material.
export function styleBody(node, index, bodyNames) {
  const material = new THREE.MeshStandardMaterial({
    color: bodyColor(index), roughness: 0.6, metalness: 0.0,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, // edges on top
  });
  for (const mesh of ownMeshes(node, bodyNames)) {
    mesh.geometry = withNormals(mesh.geometry);
    mesh.material = material;
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry, 30),
      new THREE.LineBasicMaterial({ color: EDGE_COLOR, transparent: true, opacity: 0.55 }),
    );
    edges.name = EDGE_NAME;
    edges.raycast = () => {};
    mesh.add(edges);
  }
  return material;
}

export function setHighlighted(material, on, index) {
  material.color.set(on ? HIGHLIGHT : bodyColor(index));
  material.emissive.set(on ? HIGHLIGHT_EMISSIVE : 0x000000);
}

export function addLights(scene) {
  scene.add(new THREE.HemisphereLight(0xffffff, 0x7d8794, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 2.0);
  key.position.set(3, 5, 4);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.6);
  fill.position.set(-4, 2, -3);
  scene.add(fill);
}
