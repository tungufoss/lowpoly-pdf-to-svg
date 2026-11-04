const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const html = fs.readFileSync('app/fold3d-viewer.html', 'utf8');
const start = html.indexOf('class Palette {');
const Palette = vm.runInNewContext('(' + html.slice(start, html.indexOf('\n}\n', start) + 2) + ')');
const square = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]];
const model = {
  pieces: {'piece-01-01': {material: 'red'}, 'piece-01-02': {material: 'red'}, 'piece-01-03': {material: 'red'},
           'piece-02-01': {material: 'gold'}},
  colours: {red: '#ff0000'}, vertices: square,
  faces: Object.keys({'piece-01-01': 0, 'piece-01-02': 0, 'piece-01-03': 0, 'piece-02-01': 0}).map(piece => ({piece, v: [0, 1, 2, 3]})),
  glues: [{piece: 'piece-01-01', tab_piece: 'piece-01-02'}, {piece: 'piece-01-02', tab_piece: 'piece-02-01'}],
};
model.gluesOf = p => model.glues.filter(g => g.piece === p || g.tab_piece === p);
const palette = new Palette({sheet_mm: [210, 297], materials: [
  {id: 'red', name: 'Red cardstock', color: '#c62828', pages: [1], pieces: []},
  {id: 'brown', name: 'Brown cardstock', color: '#6d4c41', pages: [], pieces: ['piece-01-03']},
  {id: 'gold', name: 'Gold cardstock', color: '#d4a017', pages: [2], pieces: []},
  {id: 'blue', name: 'Blue cardstock', color: '#1565c0', pages: [], pieces: [], spare: true}]}, model);
palette.apply(model);
// The YAML's pieces: wins over fold3d.json's material, and its colours replace fold3d.json's.
assert.equal(model.pieces['piece-01-03'].material, 'brown');
assert.equal(model.colours.red, '#c62828');
assert.equal(JSON.stringify(model.paletteIds), JSON.stringify(['red', 'brown', 'gold', 'blue']));
// Region: pieces of the clicked colour glued to it, not across colours.
assert.equal(JSON.stringify(palette.pick(model, 'piece-01-01', 'region', new Map()).sort()), JSON.stringify(['piece-01-01', 'piece-01-02']));
assert.equal(JSON.stringify(palette.pick(model, 'piece-01-01', 'twins', new Map([['piece-01-01', 'piece-01-03']]))), JSON.stringify(['piece-01-01', 'piece-01-03']));
assert.equal(palette.csv(model), 'piece,material\n');
model.pieces['piece-01-02'].material = 'blue';
model.pieces['piece-01-03'].material = 'red';
assert.equal(palette.csv(model), 'piece,material\npiece-01-02,blue\npiece-01-03,red\n');
const yaml = palette.yaml(model);
assert.match(yaml, /- id: blue\n    name: Blue cardstock\n    color: "#1565c0"\n    pieces: \[piece-01-02\]/);
assert.doesNotMatch(yaml, /piece-01-03/);   // back on its page's colour
assert.match(yaml, /- id: brown\n    name: Brown cardstock\n  - id: gold/);
assert.equal(Math.round(palette.areas(model).get('blue')), 100);
console.log('Verified the viewer palette follows the YAML, picks twins and same-colour regions, and writes recolour.csv and YAML.');
