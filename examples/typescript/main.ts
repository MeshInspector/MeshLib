import * as ml from '@meshinspector/meshlib';

await ml.init();

// Build a mesh from raw typed-array geometry.
const positions = new Float32Array([
  -1, -1, -1,  1, -1, -1,  1, 1, -1,  -1, 1, -1,
  -1, -1,  1,  1, -1,  1,  1, 1,  1,  -1, 1,  1,
]);
const indices = new Int32Array([
  0, 2, 1,  0, 3, 2,  4, 5, 6,  4, 6, 7,  0, 1, 5,  0, 5, 4,
  3, 6, 2,  3, 7, 6,  0, 4, 7,  0, 7, 3,  1, 2, 6,  1, 6, 5,
]);
const coords: ml.VertCoords = ml.VertCoords.fromArray(positions);
const tris: ml.Triangulation = ml.Triangulation.fromArray(indices);
const cube: ml.Mesh = ml.Mesh.fromTriangles(coords, tris);
coords.delete();
tris.delete();
console.log('cube volume =', cube.volume().toFixed(3));

// A maker and a boolean, with the enum of the operation.
const sphere: ml.Mesh = ml.makeUVSphere(1, 24, 24);
const res: ml.BooleanResult = ml.boolean(cube, sphere, ml.BooleanOperation.Union);
if (!res.valid())
  throw new Error(res.errorString);
const union: ml.Mesh = res.mesh;
console.log('union volume =', union.volume().toFixed(3));

// A projection query: the result is a typed struct.
const part = new ml.MeshPart(union);
const proj: ml.MeshProjectionResult = ml.findProjection(new ml.Vector3f(5, 0, 0), part);
console.log(`nearest face ${proj.proj.face.id_} at dist ${Math.sqrt(proj.distSq).toFixed(3)}`);

// The geometry back as typed arrays (Float32Array / Int32Array).
union.pack();
const outPositions: Float32Array = union.points.toArray();
const outIndices: Int32Array = union.topology.getTriangulation().toArray();
console.log(`union: ${outPositions.length / 3} verts, ${outIndices.length / 3} tris`);

part.delete();
res.delete();
sphere.delete();
cube.delete();

process.exit(0);
