import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// Load mesh
using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// Repack the mesh.
// Not necessary, but highly recommended to achieve the best performance in parallel processing.
mesh.packOptimally();

// Setup decimate parameters
using settings = new ml.DecimateSettings();

// Decimation stop thresholds, you may specify one or both
settings.maxDeletedFaces = 1000; // Number of faces to be deleted
settings.maxError = 0.05; // Maximum error when decimation stops

// Number of parts for parallel processing, greatly improves performance at the cost of minor quality loss.
// A value close to the number of CPU cores is usually the fastest; much larger values add per-part overhead
settings.subdivideParts = 64;

// Decimate mesh
ml.decimateMesh(mesh, settings);

// Save result
ml.MeshSave.toAnySupportedFormat(mesh, 'decimated_mesh.stl');
