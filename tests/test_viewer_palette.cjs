const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const html = fs.readFileSync('app/fold3d-viewer.html', 'utf8');
const start = html.indexOf('class Palette {');
const Palette = vm.runInNewContext('(' + html.slice(start, html.indexOf('\n}\n', start) + 2) + ')');
const json = v => JSON.stringify(v);
// 10 x 10 mm squares in a row about x = 0: piece-01-01 has two faces (/1, /2), 01-03 mirrors /1.
const vertices = [[-20, 0, 0], [-10, 0, 0], [0, 0, 0], [-20, 10, 0], [-10, 10, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0], [20, 0, 0], [20, 10, 0]];
const model = {
  pieces: {'piece-01-01': {material: 'red'}, 'piece-01-02': {material: 'red'}, 'piece-01-03': {material: 'red'},
           'piece-02-01': {material: 'gold'}},
  colours: {red: '#ff0000'}, vertices,
  faces: [{id: 'piece-01-01/1', piece: 'piece-01-01', v: [0, 1, 4, 3]}, {id: 'piece-01-01/2', piece: 'piece-01-01', v: [1, 2, 5, 4]},
          {id: 'piece-01-02/1', piece: 'piece-01-02', v: [2, 6, 7, 5]}, {id: 'piece-01-03/1', piece: 'piece-01-03', v: [6, 8, 9, 7]},
          {id: 'piece-02-01/1', piece: 'piece-02-01', v: [0, 1, 4, 3]}],
  glues: [],
};
const palette = new Palette({sheet_mm: [210, 297], materials: [
  {id: 'red', name: 'Red cardstock', color: '#c62828', pages: [1], pieces: []},
  {id: 'brown', name: 'Brown cardstock', color: '#6d4c41', pages: [], pieces: ['piece-01-03']},
  {id: 'gold', name: 'Gold cardstock', color: '#d4a017', pages: [2], pieces: []},
  {id: 'blue', name: 'Blue cardstock', color: '#1565c0', pages: [], pieces: [], faces: [], spare: true}]}, model);
palette.apply(model);
// The YAML's pieces: wins over fold3d.json's material, and its colours replace fold3d.json's.
assert.equal(model.pieces['piece-01-03'].material, 'brown');
assert.equal(model.colours.red, '#c62828');
assert.equal(json(model.paletteIds), json(['red', 'brown', 'gold', 'blue']));
// Region: faces of the clicked colour joined by a fold or shared edge; brown 01-03 is not.
assert.equal(json(palette.pick(model, 0, 'region', new Map()).sort()), json([0, 1, 2]));
assert.equal(json(palette.pick(model, 0, 'twins', new Map([['piece-01-01', 'piece-01-03']]))), json([0, 1, 3]));
// A face and its mirror: the twin's face nearest its reflection in x = 0 (face numbers may differ).
assert.equal(json(palette.pick(model, 0, 'face', new Map([['piece-01-01', 'piece-01-03']]))), json([0, 3]));
assert.equal(json(palette.pick(model, 1, 'face', new Map())), json([1, 2]));   // no twin: any face
assert.equal(palette.csv(model), 'piece,material\n');
// One face of 01-01 goes blue: the piece is split, the row names the face.
palette.setFaces(model, [1], 'blue');
assert.equal(model.pieces['piece-01-01'].material, 'red');
assert.equal(model.faces[1].material, 'blue');
assert.equal(json(palette.split(model)), json(['piece-01-01']));
assert.equal(palette.csv(model), 'piece,material\npiece-01-01/2,blue\n');
assert.match(palette.yaml(model), /- id: blue\n    name: Blue cardstock\n    color: "#1565c0"\n    faces: \[piece-01-01\/2\]/);
// Its other face goes blue too: the whole piece is blue, no split, one piece row.
palette.setFaces(model, [0], 'blue');
assert.equal(model.pieces['piece-01-01'].material, 'blue');
assert.equal(model.faces[0].material, undefined);
assert.equal(json(palette.split(model)), json([]));
assert.equal(palette.csv(model), 'piece,material\npiece-01-01,blue\n');
model.pieces['piece-01-03'].material = 'red';
palette.setFaces(model, [3], 'red');
assert.equal(palette.csv(model), 'piece,material\npiece-01-01,blue\npiece-01-03,red\n');
const yaml = palette.yaml(model);
assert.match(yaml, /    pieces: \[piece-01-01\]/);
assert.doesNotMatch(yaml, /piece-01-03/);   // back on its page's colour
assert.equal(Math.round(palette.areas(model).get('blue')), 200);
// A face listed under faces: in the YAML starts split.
const again = JSON.parse(json(model));
again.faces.forEach(f => delete f.material);
Object.values(again.pieces).forEach(p => { p.material = 'red'; });
const yamlFaces = new Palette({materials: [{id: 'red', name: 'Red', color: '#c62828', pages: [1]},
  {id: 'gold', name: 'Gold', color: '#d4a017', pages: [2], faces: ['piece-01-01/2']}]}, again);
yamlFaces.apply(again);
assert.equal(again.faces[1].material, 'gold');
assert.equal(json(yamlFaces.split(again)), json(['piece-01-01']));
assert.equal(yamlFaces.csv(again), 'piece,material\n');
console.log('Verified the viewer palette follows the YAML, picks faces, twins and regions, splits pieces by face colour, and writes recolour.csv and YAML.');
