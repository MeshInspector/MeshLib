import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

using meshA = ml.makeUVSphere(1.0, 16, 16); // make mesh A
using meshB = ml.makeUVSphere(1.0, 16, 16); // make mesh B
// shift mesh B for better demonstration
using shift = ml.AffineXf3f.translation({ x: 0.1, y: 0.1, z: 0.1 });
meshB.transform(shift);

// create converters to integer field (needed for absolute precision predicates)
using converters = ml.getVectorConverters(meshA, meshB);
// find each intersecting edge/triangle pair
using collidingFaceEdges = ml.findCollidingEdgeTrisPrecise(meshA, meshB, converters);
for (let i = 0; i < collidingFaceEdges.size(); ++i) {
  using vet = collidingFaceEdges.get(i);
  if (vet.isEdgeATriB())
    console.log(`edgeA: ${vet.edge}, triB: ${vet.tri()}`);
  else
    console.log(`triA: ${vet.tri()}, edgeB: ${vet.edge}`);
}
