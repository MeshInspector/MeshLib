import * as ml from '@meshinspector/meshlib';

await ml.init();

// create some mesh
using mesh = ml.makeCube(new ml.Vector3f(1, 1, 1), new ml.Vector3f(-0.5, -0.5, -0.5));

// all point coordinates, as a flat Float32Array of x, y, z triples
const vertexCoordinates = mesh.points.toArray();

// triangle vertices, as a flat Int32Array of triples of indices into the points array
const vertexTriples = mesh.topology.getTriangulation().toArray();

// TODO: export in your format
console.log(`${vertexCoordinates.length / 3} vertices, ${vertexTriples.length / 3} triangles`);
