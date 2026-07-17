import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// Load mesh
using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// Repack the mesh.
// Not necessary, but highly recommended to achieve the best performance in parallel processing.
mesh.pack();

// Setup decimate parameters
using settings = new ml.DecimateSettings();

// Decimation stop thresholds, you may specify one or both
settings.maxDeletedFaces = 1000; // Number of faces to be deleted
settings.maxError = 0.05; // Maximum error when decimation stops

// Number of parts for simultaneous processing, greatly improves performance by cost of minor quality loss
settings.subdivideParts = 64;

// Decimate mesh
ml.decimateMesh(mesh, settings);

// Save result
ml.MeshSave.toAnySupportedFormat(mesh, 'decimated_mesh.stl');
