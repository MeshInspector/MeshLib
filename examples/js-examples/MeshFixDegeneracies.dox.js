import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// you can set various parameters for the resolving process; see the documentation for more info
using box = mesh.computeBoundingBox();
using params = new ml.FixMeshDegeneraciesParams();
params.maxDeviation = 1e-5 * box.diagonal();
params.tinyEdgeLength = 1e-3;
ml.fixMeshDegeneracies(mesh, params);

// Save result
ml.MeshSave.toAnySupportedFormat(mesh, 'fixedMesh.stl');
