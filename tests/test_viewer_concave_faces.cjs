const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');const html=fs.readFileSync('app/fold3d-viewer.html','utf8'),ctx=vm.createContext({});const start=html.indexOf('function faceTriangles(');vm.runInContext(html.slice(start,html.indexOf('function faceOutlineEdges(',start)),ctx);
const flat=[[1283.28,1154.22],[1270.63,1133.38],[1265.27,1099.94],[1233.85,1114.41],[1253.71,1150.45]];
const area=p=>Math.abs(p.reduce((s,a,i)=>{const b=p[(i+1)%p.length];return s+a[0]*b[1]-a[1]*b[0]},0))/2;
for(const points of [flat,[...flat].reverse()]){const triangles=ctx.faceTriangles({flat:points,id:'concave-regression'});assert.equal(triangles.length,3);assert.ok(Math.abs(triangles.reduce((s,t)=>s+area(t.map(i=>points[i])),0)-area(points))<1e-6);}
console.log('Verified concave face triangulation covers exactly the paper polygon, in both winding directions.');
