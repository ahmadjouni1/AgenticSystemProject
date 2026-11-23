// Forward kinematics for tree mechanisms, mirroring sdk/kinematics.py.
// Matrices are row-major arrays of 16 numbers; Z-up; meters and radians.
// No dependencies, so the tests can run it under Node and compare it with Python.

export function identity() {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

export function multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[r * 4 + k] * b[k * 4 + c];
      out[r * 4 + c] = sum;
    }
  }
  return out;
}

// Inverse of a rigid transform: rotation transposed, translation -R^T t.
export function invertRigid(m) {
  const out = identity();
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) out[r * 4 + c] = m[c * 4 + r];
  }
  for (let r = 0; r < 3; r++) {
    out[r * 4 + 3] = -(out[r * 4] * m[3] + out[r * 4 + 1] * m[7] + out[r * 4 + 2] * m[11]);
  }
  return out;
}

// URDF roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll).
export function rpyMatrix([roll, pitch, yaw]) {
  const [cr, sr, cp, sp, cy, sy] = [Math.cos(roll), Math.sin(roll), Math.cos(pitch),
    Math.sin(pitch), Math.cos(yaw), Math.sin(yaw)];
  return [
    cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
    -sp, cp * sr, cp * cr,
  ];
}

function transform(rotation, translation) {
  const [r, t] = [rotation, translation];
  return [r[0], r[1], r[2], t[0], r[3], r[4], r[5], t[1], r[6], r[7], r[8], t[2], 0, 0, 0, 1];
}

export function frameMatrix(frame) {
  return transform(rpyMatrix(frame.rpy), frame.xyz);
}

// Rotation by angle about a unit axis (Rodrigues), as sdk.transforms.axis_angle_matrix.
function axisAngle([x, y, z], angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const k = [0, -z, y, z, 0, -x, -y, x, 0];
  const kk = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) {
    for (let col = 0; col < 3; col++) {
      for (let i = 0; i < 3; i++) kk[r * 3 + col] += k[r * 3 + i] * k[i * 3 + col];
    }
  }
  return [0, 1, 2, 3, 4, 5, 6, 7, 8].map(
    (i) => (i % 4 === 0 ? 1 : 0) + s * k[i] + (1 - c) * kk[i],
  );
}

export function motion(joint, value) {
  if (joint.type === "revolute") return transform(axisAngle(joint.axis, value), [0, 0, 0]);
  if (joint.type === "prismatic") {
    return transform([1, 0, 0, 0, 1, 0, 0, 0, 1], joint.axis.map((a) => a * value));
  }
  return identity();
}

// Every joint value: driven ones as given (others at 0, the rest pose), then mimics.
export function jointValues(manifest, driven) {
  const values = {};
  for (const joint of manifest.joints) {
    if (joint.role === "tree" && joint.type !== "fixed" && !joint.mimic) {
      values[joint.name] = driven[joint.name] ?? 0;
    }
  }
  for (const m of manifest.mimics) values[m.joint] = m.multiplier * values[m.source] + m.offset;
  return values;
}

// World transform of every body, placed down the joint tree from the root.
export function bodyTransforms(manifest, driven) {
  const values = jointValues(manifest, driven);
  const joints = Object.fromEntries(manifest.joints.map((j) => [j.name, j]));
  const world = { [manifest.articulation.root]: identity() };
  for (const name of manifest.articulation.tree) {
    const joint = joints[name];
    world[joint.child] = multiply(
      multiply(multiply(world[joint.parent], frameMatrix(joint.parent_frame)),
        motion(joint, values[name] ?? 0)),
      invertRigid(frameMatrix(joint.child_frame)),
    );
  }
  return world;
}

// Each body's matrix relative to its parent body (the GLB node hierarchy).
export function localMatrices(manifest, world) {
  const joints = Object.fromEntries(manifest.joints.map((j) => [j.name, j]));
  const local = { [manifest.articulation.root]: world[manifest.articulation.root] };
  for (const name of manifest.articulation.tree) {
    const joint = joints[name];
    local[joint.child] = multiply(invertRigid(world[joint.parent]), world[joint.child]);
  }
  return local;
}
