import * as ml from '@meshinspector/meshlib';

await ml.init();

// Load mesh
using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// Compute mesh bounding box
using box = mesh.computeBoundingBox();

// Construct the deformer on the mesh vertices (it references the mesh, which is declared first so it outlives the deformer)
using ffDeformer = new ml.FreeFormDeformer(mesh.points, mesh.topology.getValidVerts());

// Init deformer with a 3x3x3 grid on the mesh box
ffDeformer.init(ml.Vector3i.diagonal(3), box);

// Move some control points of the grid to the center
const center = box.center();
ffDeformer.setRefGridPointPosition(new ml.Vector3i(1, 1, 0), center);
ffDeformer.setRefGridPointPosition(new ml.Vector3i(1, 1, 2), center);
ffDeformer.setRefGridPointPosition(new ml.Vector3i(0, 1, 1), center);
ffDeformer.setRefGridPointPosition(new ml.Vector3i(2, 1, 1), center);
ffDeformer.setRefGridPointPosition(new ml.Vector3i(1, 0, 1), center);
ffDeformer.setRefGridPointPosition(new ml.Vector3i(1, 2, 1), center);

// Apply the deformation to the mesh vertices
ffDeformer.apply();

// Invalidate the mesh because of the external vertex changes
mesh.invalidateCaches();

// Save deformed mesh
ml.MeshSave.toAnySupportedFormat(mesh, 'deformed_mesh.stl');
