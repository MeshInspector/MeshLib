import createMeshLib, { type Mesh } from '@meshinspector/meshlib';

const ml = await createMeshLib();

// Build a mesh from raw typed-array geometry (typed converters: Float32Array/Uint32Array).
const positions = new Float32Array([
  -1, -1, -1,  1, -1, -1,  1, 1, -1,  -1, 1, -1,
  -1, -1,  1,  1, -1,  1,  1, 1,  1,  -1, 1,  1,
]);
const indices = new Uint32Array([
  0, 2, 1,  0, 3, 2,  4, 5, 6,  4, 6, 7,  0, 1, 5,  0, 5, 4,
  3, 6, 2,  3, 7, 6,  0, 4, 7,  0, 7, 3,  1, 2, 6,  1, 6, 5,
]);
const coords = ml.VertCoords.fromArray(positions);
const tris = ml.Triangulation.fromArray(indices);
const cube: Mesh = ml.Mesh.fromTriangles(coords, tris)!;
coords.delete();
tris.delete();
console.log('cube volume =', cube.volume().toFixed(3));

// A generated maker + boolean (generated enum value).
const sphere: Mesh = ml.makeUVSphere(1, 24, 24)!;
const res = ml.boolean(cube, sphere, ml.BooleanOperation.Union);
if (!res.valid())
  throw new Error(res.errorString);
const union: Mesh = res.mesh!;
console.log('union volume =', union.volume().toFixed(3));

// Overlay-refined val return: a precisely-typed result object.
const proj = ml.findProjection({ x: 5, y: 0, z: 0 }, union);
console.log(`nearest face ${proj.proj.face} at dist ${Math.sqrt(proj.distSq).toFixed(3)} (valid=${proj.valid})`);

// Overlay-refined val out: typed toArray (Float32Array / Uint32Array), not `any`.
union.pack();
const pts = union.points;
const topo = union.topology;
const tri = topo.getTriangulation();
const outPositions: Float32Array = pts.toArray();
const outIndices: Uint32Array = tri.toArray();
console.log(`union: ${outPositions.length / 3} verts, ${outIndices.length / 3} tris`);

pts.delete();
topo.delete();
tri.delete();
union.delete();
res.delete();
sphere.delete();
cube.delete();

process.exit(0);
