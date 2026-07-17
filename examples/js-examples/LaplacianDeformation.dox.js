import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// Load mesh
using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// Construct deformer on the mesh vertices (it references the mesh, which is declared first so it
// outlives the deformer)
using deformer = new ml.Laplacian(mesh);

// Find an area for the deformation anchor points
using topology = mesh.topology;
using validVerts = topology.getValidVerts();
const ancV0 = validVerts.find_first();
const ancV1 = validVerts.find_last();

// Mark the anchor points in the free area
using freeVerts = new ml.VertBitSet();
freeVerts.resize(validVerts.size(), false);
freeVerts.set(ancV0, true);
freeVerts.set(ancV1, true);
// Expand the free area
using expandedFreeVerts = ml.expandVerts(topology, freeVerts, 5);

// Initialize laplacian
deformer.init(expandedFreeVerts, ml.EdgeWeights.Cotan, ml.VertexMass.NeiArea, ml.RememberShape.Yes);

using box = mesh.computeBoundingBox();
const shiftAmount = box.diagonal() * 0.01;
using points = mesh.points;
// Fix the anchor vertices in the required position (shifted along the vertex normal)
for (const anchor of [ancV0, ancV1]) {
  const p = points.get(anchor);
  const n = mesh.normal(anchor);
  deformer.fixVertex(
    anchor,
    { x: p.x + n.x * shiftAmount, y: p.y + n.y * shiftAmount, z: p.z + n.z * shiftAmount },
    true );
}

// Move the free vertices according to the anchor ones
deformer.apply();

// Invalidate the mesh because of the external vertex changes
mesh.invalidateCaches();

// Save the deformed mesh
ml.MeshSave.toAnySupportedFormat(mesh, 'deformed_mesh.stl');
