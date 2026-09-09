import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

using meshA = ml.makeUVSphere(1.0, 16, 16); // make mesh A
using meshB = ml.makeUVSphere(1.0, 16, 16); // make mesh B
// shift mesh B for better demonstration
using shift = ml.AffineXf3f.translation({ x: 0.1, y: 0.1, z: 0.1 });
meshB.transform(shift);

// find each pair of colliding faces
const collidingFacePairs = ml.findCollidingTriangles(meshA, meshB);
for (const { aFace, bFace } of collidingFacePairs)
  console.log(`${aFace} ${bFace}`);

// find bitsets of colliding faces
const bitsets = ml.findCollidingTriangleBitsets(meshA, meshB);
using collidingFacesA = bitsets.a;
using collidingFacesB = bitsets.b;
console.log(collidingFacesA.count()); // number of colliding faces from mesh A
console.log(collidingFacesB.count()); // number of colliding faces from mesh B

// fast check whether mesh A and mesh B collide
const isColliding = ml.findCollidingTriangles(meshA, meshB, true).length > 0;
console.log(isColliding);
