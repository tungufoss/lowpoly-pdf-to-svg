const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(path.join(__dirname, '../app/fold3d-viewer.html'), 'utf8');
const start = html.indexOf('  focus(face, onlyIfOffCenter');
const method = html.slice(start, html.indexOf('  focusPiece(', start));
class Vector3 {
  constructor(x = 0, y = 0, z = 0) { Object.assign(this, {x, y, z}); }
  clone() { return new Vector3(this.x, this.y, this.z); }
  add(v) { this.x += v.x; this.y += v.y; this.z += v.z; return this; }
  sub(v) { this.x -= v.x; this.y -= v.y; this.z -= v.z; return this; }
  divideScalar(n) { this.x /= n; this.y /= n; this.z /= n; return this; }
  copy(v) { Object.assign(this, v); return this; }
  // Synthetic world coordinates are normalized pane coordinates in these tests.
  project() { return this; }
}
const focus = vm.runInNewContext(`({${method}}).focus`, {THREE: {Vector3}});
function view(x, y) {
  return {model: {faces: [{v: [0]}], vertices: [[x, y, 0]]},
    camera: {position: new Vector3(0, 0, 10), zoom: 2, quaternion: 'same orientation'},
    controls: {target: new Vector3(), update() {}}};
}
for (const [x, y, moves] of [[0.2, 0.2, false], [0.4, 0.4, false], [0.6, 0, true], [0, -0.6, true]]) {
  const v = view(x, y);
  focus.call(v, 0, true);
  assert.equal(v.controls.target.x, moves ? x : 0);
  assert.equal(v.controls.target.y, moves ? y : 0);
  assert.deepEqual(v.camera.position.clone().sub(v.controls.target), new Vector3(0, 0, 10));
  assert.equal(v.camera.zoom, 2);
  assert.equal(v.camera.quaternion, 'same orientation');
}
assert.match(html, /this\.select\(face, "3d", false\)/);
console.log('Verified central clicks stay put; outer clicks pan without changing camera offset or zoom.');
