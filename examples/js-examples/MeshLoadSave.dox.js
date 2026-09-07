import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// Load mesh
using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// Save mesh (the format is chosen from the file extension)
ml.MeshSave.toAnySupportedFormat(mesh, 'mesh.ply');
