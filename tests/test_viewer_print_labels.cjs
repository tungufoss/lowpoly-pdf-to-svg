const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(path.join(__dirname, '../app/fold3d-viewer.html'), 'utf8');
const moduleText = html.split('<script type="module">')[1].split('</script>')[0];
// Parse the complete module, independently of the browser's import map.
new (Object.getPrototypeOf(async function () {}).constructor)(moduleText.replace(/^import .*;$/gm, ''));
const start = html.indexOf('function readPrintLabels(');
const code = html.slice(start, html.indexOf('function flatBarycentric(', start));
function text(value, x, y, angle, pieceNumber = false) {
  const attrs = {x, y, transform: `rotate(${angle} ${x} ${y})`, 'font-size': '5'};
  return {textContent: value, getAttribute: name => attrs[name], hasAttribute: () => pieceNumber,
    parentNode: {querySelector: () => ({getAttribute: () => 'circle'})}};
}
const source = text('123', '10', '20', '0');
const edge = text('AB', '40', '50', '90');
const number = text('7', '45', '55', '30', true);
const doc = nodes => ({querySelectorAll: () => [{id: 'piece-01-01', querySelectorAll: () => nodes}]});
const context = vm.createContext({DOMParser: class {parseFromString(s) {return doc(s === 'export' ? [edge, number] : [source]);}},
  pieceFlatOffset: () => ({x: 2, y: 3}), THREE: {MathUtils: {degToRad: n => n * Math.PI / 180}}});
vm.runInContext(code, context);
const labels = context.readPrintLabels('source', [{pieces: 'piece-01-01', pdf_number: '123', label: 'AB'}], {faces: []}, 'export');
assert.equal(labels.length, 3);
assert.equal(labels[0].cricut, '');
assert.equal(labels[1].cricut, 'AB');
assert.equal(labels[1].x, 42);
assert.equal(labels[1].y, 53);
assert.equal(labels[1].angle, Math.PI / 2);
assert.equal(labels[2].cricut, '○7');
assert.match(html, /label\.x - Math\.sin\(label\.angle\), label\.y \+ Math\.cos\(label\.angle\)/);
console.log('Verified module syntax, export label positions/angles, piece markers, and rotated label axes.');
