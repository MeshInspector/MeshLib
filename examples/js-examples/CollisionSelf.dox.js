import * as ml from '@meshinspector/meshlib';

await ml.init();

using mesh = ml.makeTorusWithSelfIntersections(); // make torus with self-intersections
using part = new ml.MeshPart(mesh);

// find self-intersecting face pairs (throws if the search is cancelled)
using selfCollidingPairs = ml.findSelfCollidingTriangles(part);
for (let i = 0; i < selfCollidingPairs.size(); ++i) {
  const fp = selfCollidingPairs.at(i);
  console.log(fp.aFace.id_, fp.bFace.id_); // print each pair
}

// find the union of self-intersecting faces
using selfCollidingFaces = ml.findSelfCollidingTrianglesBS(part);
console.log(selfCollidingFaces.count()); // number of self-intersecting faces
