import * as ml from '@meshinspector/meshlib';

await ml.init();

using meshA = ml.makeUVSphere(); // make mesh A
using meshB = ml.makeUVSphere(); // make mesh B
meshB.transform(ml.AffineXf3f.translation(new ml.Vector3f(0.1, 0.1, 0.1))); // shift mesh B for better demonstration

using partA = new ml.MeshPart(meshA);
using partB = new ml.MeshPart(meshB);

// create converters to integer field (needed for absolute precision predicates)
using converters = ml.getVectorConverters(partA, partB);

// find each intersecting edge/triangle pair
using collidingFaceEdges = ml.findCollidingEdgeTrisPrecise(partA, partB, converters.toInt);
for (let i = 0; i < collidingFaceEdges.size(); ++i) {
  const vet = collidingFaceEdges.index(i);
  if (vet.isEdgeATriB())
    console.log(`edgeA: ${vet.edge.id_}, triB: ${vet.tri().id_}`);
  else
    console.log(`triA: ${vet.tri().id_}, edgeB: ${vet.edge.id_}`);
}
