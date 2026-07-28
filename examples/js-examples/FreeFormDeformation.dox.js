import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// Load mesh
using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// Construct the deformer on the mesh (the JS binding takes the mesh directly)
using ffDeformer = new ml.FreeFormDeformer(mesh);

// Compute mesh bounding box
using box = mesh.computeBoundingBox();

// Init deformer with a 3x3x3 grid on the mesh box
ffDeformer.init({ x: 3, y: 3, z: 3 }, box);

// Move some control points of the grid to the center
const center = box.center();
ffDeformer.setRefGridPointPosition({ x: 1, y: 1, z: 0 }, center);
ffDeformer.setRefGridPointPosition({ x: 1, y: 1, z: 2 }, center);
ffDeformer.setRefGridPointPosition({ x: 0, y: 1, z: 1 }, center);
ffDeformer.setRefGridPointPosition({ x: 2, y: 1, z: 1 }, center);
ffDeformer.setRefGridPointPosition({ x: 1, y: 0, z: 1 }, center);
ffDeformer.setRefGridPointPosition({ x: 1, y: 2, z: 1 }, center);

// Apply the deformation to the mesh vertices
ffDeformer.apply();

// Invalidate the mesh because of the external vertex changes
mesh.invalidateCaches();

// Save deformed mesh
ml.MeshSave.toAnySupportedFormat(mesh, 'deformed_mesh.stl');
