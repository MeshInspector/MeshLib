import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// select the faces to extrude
using facesToExtrude = ml.FaceBitSet.fromIndices(new Uint32Array([1, 2]));

// create duplicated vertices along the region boundary
ml.makeDegenerateBandAroundRegion(mesh, facesToExtrude);

// find the vertices to move and shift them along +Z
using topology = mesh.topology;
using vertsForMove = ml.getIncidentVertsFromFaces(topology, facesToExtrude);
using coords = mesh.points;
for (const v of vertsForMove.toIndices()) {
  const p = coords.get(v);
  p.z += 1;
  coords.set(v, p);
}
mesh.points = coords;

// invalidate internal caches after the manual vertex changes
mesh.invalidateCaches();

ml.MeshSave.toAnySupportedFormat(mesh, 'extrudedMesh.stl');
