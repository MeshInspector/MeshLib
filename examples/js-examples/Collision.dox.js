import * as ml from '@meshinspector/meshlib';

await ml.init();

using meshA = ml.makeUVSphere(); // make mesh A
using meshB = ml.makeUVSphere(); // make mesh B
meshB.transform(ml.AffineXf3f.translation(new ml.Vector3f(0.1, 0.1, 0.1))); // shift mesh B for better demonstration

// the collision functions take mesh parts (a mesh plus an optional region)
using partA = new ml.MeshPart(meshA);
using partB = new ml.MeshPart(meshB);

// find each pair of colliding faces
using collidingFacePairs = ml.findCollidingTriangles(partA, partB);
for (let i = 0; i < collidingFacePairs.size(); ++i) {
  const fp = collidingFacePairs.at(i);
  console.log(fp.aFace.id_, fp.bFace.id_); // print each pair of colliding faces
}

// find bitsets of colliding faces
using collidingBitSets = ml.findCollidingTriangleBitsets(partA, partB);
console.log(collidingBitSets.first().count()); // number of colliding faces from mesh A
console.log(collidingBitSets.second().count()); // number of colliding faces from mesh B

// fast check whether mesh A and mesh B collide
using firstPair = ml.findCollidingTriangles(partA, partB, null, true);
const isColliding = !firstPair.empty();
console.log(isColliding);
