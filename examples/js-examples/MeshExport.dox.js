import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// create some mesh
using mesh = ml.makeCube({ x: 1, y: 1, z: 1 }, { x: -0.5, y: -0.5, z: -0.5 });

// all point coordinates, as a flat Float32Array of x, y, z triples
using points = mesh.points;
const vertexCoordinates = points.toArray();

// triangle vertices, as a flat Uint32Array of triples of indices into the points array
using topology = mesh.topology;
using triangulation = topology.getTriangulation();
const vertexTriples = triangulation.toArray();

// TODO: export in your format
console.log(`${vertexCoordinates.length / 3} vertices, ${vertexTriples.length / 3} triangles`);
