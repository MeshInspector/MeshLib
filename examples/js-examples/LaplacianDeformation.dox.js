import * as ml from '@meshinspector/meshlib';

await ml.init();

// Load mesh
using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// Construct the deformer on the mesh vertices (it references the mesh, which is declared first so it outlives the deformer)
using deformer = new ml.Laplacian(mesh);

// Find an area for the deformation anchor points
const validVerts = mesh.topology.getValidVerts();
const ancV0 = validVerts.find_first();
const ancV1 = validVerts.find_last();

// Mark the anchor points in the free area
using freeVerts = new ml.VertBitSet();
freeVerts.resize(validVerts.size(), false);
freeVerts.set(ancV0, true);
freeVerts.set(ancV1, true);

// Expand the free area
ml.expand(mesh.topology, freeVerts, 5);

// Initialize laplacian
deformer.init(freeVerts, ml.EdgeWeights.Cotan, ml.VertexMass.NeiArea, ml.RememberShape.Yes);

const shiftAmount = mesh.computeBoundingBox().diagonal() * 0.01;

// Fix the anchor vertices in the required position (shifted along the vertex normal)
for (const anchor of [ancV0, ancV1]) {
  const p = mesh.points.index(anchor);
  const n = mesh.normal(anchor);
  deformer.fixVertex(anchor, ml.opAdd(p, ml.opMul(n, shiftAmount)), true);
}

// Move the free vertices according to the anchor ones
deformer.apply();

// Invalidate the mesh because of the external vertex changes
mesh.invalidateCaches();

// Save the deformed mesh
ml.MeshSave.toAnySupportedFormat(mesh, 'deformed_mesh.stl');
