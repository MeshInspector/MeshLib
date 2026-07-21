import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

using mesh = ml.makeTorusWithSelfIntersections(1.0, 0.1, 16, 16); // make torus with self-intersections

// find self-intersecting face pairs (throws if the search is cancelled)
const selfCollidingPairs = ml.findSelfCollidingTriangles(mesh);
for (const { aFace, bFace } of selfCollidingPairs)
  console.log(`${aFace} ${bFace}`);

// find the union of self-intersecting faces
using selfCollidingFaces = ml.findSelfCollidingTrianglesBS(mesh);
console.log(selfCollidingFaces.count()); // number of self-intersecting faces
