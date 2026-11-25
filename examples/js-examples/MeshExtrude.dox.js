import * as ml from '@meshinspector/meshlib';

await ml.init();

// Load mesh
using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// Prepare region to extrude
using facesToExtrude = new ml.FaceBitSet();
facesToExtrude.resize(3, false);
facesToExtrude.set(new ml.FaceId(1), true);
facesToExtrude.set(new ml.FaceId(2), true);

// Create duplicated verts on region boundary
ml.makeDegenerateBandAroundRegion(mesh, facesToExtrude);

// Find vertices that will be moved
using vertsForMove = ml.getIncidentVerts(mesh.topology, facesToExtrude);

// Move each vertex (the points are accessed in place)
for (let v = 0; v < vertsForMove.size(); ++v) {
  const id = new ml.VertId(v);
  if (vertsForMove.test(id))
    mesh.points.index_mut(id).z += 1.0;
}

// Invalidate internal caches after manual changing
mesh.invalidateCaches();

// Save mesh
ml.MeshSave.toAnySupportedFormat(mesh, 'extrudedMesh.stl');
