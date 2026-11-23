// The viewer page: load a version, pose it with sliders, play joints, and show compile
// signals at the pose where they happen.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { bodyTransforms, localMatrices } from "./kinematics.js";
import { addLights, bodyColor, setHighlighted, styleBody } from "./display.js";

const $ = (id) => document.getElementById(id);
const state = {
  version: null, data: null, values: {}, nodes: {}, materials: {},
  highlight: new Set(), busy: false, pending: false, playing: false,
};

// ---- three.js scene ---------------------------------------------------------------------

const renderer = new THREE.WebGLRenderer({ antialias: true, canvas: $("view") });
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf3f4f6);
const camera = new THREE.PerspectiveCamera(40, 1, 0.001, 1000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
addLights(scene);
let model = null;
let grid = null;

function resize() {
  const area = renderer.domElement.parentElement;
  renderer.setSize(area.clientWidth, area.clientHeight, false);
  camera.aspect = area.clientWidth / Math.max(1, area.clientHeight);
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

function fitCamera() {
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3()).length() || 1;
  const center = box.getCenter(new THREE.Vector3());
  camera.near = size / 1000;
  camera.far = size * 100;
  camera.position.copy(center).add(new THREE.Vector3(0.9, 0.7, 1.1).multiplyScalar(size));
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  if (grid) scene.remove(grid);
  grid = new THREE.GridHelper(size * 3, 30, 0xb8bec8, 0xdde1e6);
  grid.position.set(center.x, box.min.y, center.z);
  scene.add(grid);
}

// ---- data ---------------------------------------------------------------------------------

async function getJSON(url, options) {
  const response = await fetch(url, options);
  return { ok: response.ok, body: await response.json() };
}

function status(text, error = false) {
  $("status").textContent = text;
  $("status").className = error ? "error" : "";
}

async function init() {
  resize();
  const { body } = await getJSON("/api/versions");
  $("title").textContent = body.title;
  const select = $("versions");
  for (const v of body.versions) {
    const option = document.createElement("option");
    option.value = v.id;
    option.textContent = v.label + (v.signals ? ` (${v.signals} signals)` : "");
    select.append(option);
  }
  select.addEventListener("change", () => loadVersion(select.value));
  $("play").addEventListener("click", play);
  $("reset").addEventListener("click", () => setValues({}));
  if (body.versions.length) await loadVersion(body.versions[0].id);
  else status("This run has no exported version yet.", true);
}

async function loadVersion(id) {
  stopPlay();
  status("Loading…");
  const { body } = await getJSON(`/api/version/${id}`);
  const gltf = await new GLTFLoader().loadAsync(`/api/version/${id}/model.glb`);
  Object.assign(state, { version: id, data: body, values: {}, nodes: {}, materials: {} });
  state.highlight.clear();
  if (model) scene.remove(model);
  model = gltf.scene;
  scene.add(model);
  const names = body.manifest.bodies.map((b) => b.name);
  names.forEach((name, i) => {
    const node = model.getObjectByName(name);
    if (!node) return;
    node.matrixAutoUpdate = false;
    state.nodes[name] = node;
    state.materials[name] = { material: styleBody(node, i, names), index: i };
  });
  buildLegend(names);
  for (const d of body.drivers) state.values[d.name] = 0;
  $("mode").textContent = body.mode === "loop"
    ? "Loop mechanism: each pose is solved by the server."
    : "Tree mechanism: posed in the browser.";
  buildSliders();
  buildSignals();
  await pose();
  fitCamera();
  status("");
}

// Body name -> colour, over the 3D view. A highlighted body shows red here too.
function buildLegend(names) {
  const legend = $("legend");
  legend.innerHTML = "";
  for (const name of names) {
    if (!state.materials[name]) continue;
    const row = document.createElement("li");
    row.dataset.body = name;
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = `#${bodyColor(state.materials[name].index).toString(16).padStart(6, "0")}`;
    const label = document.createElement("span");
    label.textContent = name;
    row.append(swatch, label);
    legend.append(row);
  }
}

// ---- posing ---------------------------------------------------------------------------------

function applyWorld(world) {
  const local = localMatrices(state.data.manifest, world);
  for (const [name, matrix] of Object.entries(local)) {
    const node = state.nodes[name];
    if (!node) continue;
    node.matrix.set(...matrix);
    node.matrixWorldNeedsUpdate = true;
  }
}

// Tree: forward kinematics right here. Loop: ask the server (one request in flight; the
// newest slider values win). An unreachable loop pose keeps the last good pose.
async function pose() {
  if (state.data.mode === "tree") {
    applyWorld(bodyTransforms(state.data.manifest, state.values));
    return true;
  }
  if (state.busy) {
    state.pending = true;
    return true;
  }
  state.busy = true;
  let reached = true;
  try {
    do {
      state.pending = false;
      const { ok, body } = await getJSON(`/api/version/${state.version}/pose`, {
        method: "POST", body: JSON.stringify(state.values),
      });
      reached = ok;
      if (ok) {
        applyWorld(body.bodies);
        if ($("status").className === "error") status("");
      } else {
        status(`Cannot reach this pose (kept the last good one): ${body.error}`, true);
      }
    } while (state.pending);
  } finally {
    state.busy = false;
  }
  return reached;
}

function format(driver, value) {
  return driver.type === "revolute"
    ? `${((value * 180) / Math.PI).toFixed(1)}°`
    : `${(value * 1000).toFixed(1)} mm`;
}

function buildSliders() {
  const box = $("sliders");
  box.innerHTML = "";
  for (const driver of state.data.drivers) {
    const row = document.createElement("label");
    row.className = "slider";
    const name = document.createElement("span");
    name.textContent = driver.name;
    const input = document.createElement("input");
    Object.assign(input, {
      type: "range", min: driver.lower, max: driver.upper, value: 0,
      step: (driver.upper - driver.lower) / 600,
    });
    input.dataset.joint = driver.name;
    const value = document.createElement("span");
    value.className = "value";
    value.textContent = format(driver, 0);
    input.addEventListener("input", () => {
      state.values[driver.name] = parseFloat(input.value);
      value.textContent = format(driver, state.values[driver.name]);
      clearHighlight();
      pose();
    });
    row.append(name, input, value);
    box.append(row);
  }
  const mimics = state.data.manifest.mimics;
  $("mimics").textContent = mimics.length
    ? "Follow their source: " + mimics.map((m) => `${m.joint} = ${m.multiplier} × ${m.source}`
      + (m.offset ? ` + ${m.offset}` : "")).join("; ")
    : "";
}

function syncSliders() {
  for (const input of document.querySelectorAll("#sliders input")) {
    const driver = state.data.drivers.find((d) => d.name === input.dataset.joint);
    input.value = state.values[driver.name];
    input.nextElementSibling.textContent = format(driver, state.values[driver.name]);
  }
}

async function setValues(values) {
  stopPlay();
  clearHighlight();
  for (const d of state.data.drivers) state.values[d.name] = values[d.name] ?? 0;
  syncSliders();
  return pose();
}

// ---- play ----------------------------------------------------------------------------------

const nextFrame = () => new Promise((resolve) => requestAnimationFrame(resolve));

// Each driver in turn: rest -> upper limit -> lower limit -> rest (a loop moves through
// its driver, the solver moving the rest).
async function play() {
  if (state.playing) return stopPlay();
  state.playing = true;
  $("play").textContent = "Stop";
  clearHighlight();
  for (const driver of state.data.drivers) {
    const path = [0, driver.upper, driver.lower, 0];
    for (let leg = 0; leg < 3 && state.playing; leg++) {
      const [a, b] = [path[leg], path[leg + 1]];
      for (let f = 1; f <= 60 && state.playing; f++) {
        const eased = 0.5 - 0.5 * Math.cos((Math.PI * f) / 60);
        state.values[driver.name] = a + (b - a) * eased;
        syncSliders();
        await pose();
        await nextFrame();
      }
    }
  }
  stopPlay();
}

function stopPlay() {
  state.playing = false;
  $("play").textContent = "Play";
}

// ---- compile signals -------------------------------------------------------------------------

function buildSignals() {
  const list = $("signals");
  list.innerHTML = "";
  const signals = state.data.signals;
  $("signal-count").textContent = signals.length ? String(signals.length) : "none";
  for (const signal of signals) {
    const item = document.createElement("li");
    item.className = `signal ${signal.severity}`;
    const head = document.createElement("div");
    head.className = "head";
    head.textContent = `${signal.severity} · ${signal.kind}`;
    const text = document.createElement("div");
    text.textContent = signal.message;
    item.append(head, text);
    if (signal.pose || (signal.bodies && signal.bodies.length)) {
      item.classList.add("clickable");
      item.title = "Show this pose";
      item.addEventListener("click", () => showSignal(signal, item));
    }
    list.append(item);
  }
}

async function showSignal(signal, item) {
  for (const el of document.querySelectorAll(".signal.active")) el.classList.remove("active");
  item.classList.add("active");
  await setValues(signal.pose || {});
  setHighlight(signal.bodies || []);
  status(signal.message, signal.severity === "failure");
}

function setHighlight(bodies) {
  clearHighlight();
  for (const name of bodies) {
    const entry = state.materials[name];
    if (!entry) continue;
    setHighlighted(entry.material, true, entry.index);
    state.highlight.add(name);
  }
  markLegend();
}

function clearHighlight() {
  for (const name of state.highlight) {
    const entry = state.materials[name];
    setHighlighted(entry.material, false, entry.index);
  }
  state.highlight.clear();
  markLegend();
}

function markLegend() {
  for (const row of document.querySelectorAll("#legend li")) {
    row.classList.toggle("hit", state.highlight.has(row.dataset.body));
  }
}

init().catch((error) => status(`Viewer error: ${error.message}`, true));
