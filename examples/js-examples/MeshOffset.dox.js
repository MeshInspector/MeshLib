import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// Create mesh
using mesh = ml.makeCube({ x: 1, y: 1, z: 1 }, { x: -0.5, y: -0.5, z: -0.5 });

// Setup parameters
using params = new ml.GeneralOffsetParameters();
// calculate voxel size depending on desired accuracy and/or memory consumption
params.voxelSize = ml.suggestVoxelSize(mesh, 10000000.0);
using topology = mesh.topology;
if (ml.findRightBoundary(topology).length > 0)
  params.signDetectionMode = ml.SignDetectionMode.HoleWindingRule; // use if you have holes in mesh

// Make offset mesh
using box = mesh.computeBoundingBox();
const offset = box.diagonal() * 0.1;
using result = ml.generalOffsetMesh(mesh, offset, params);

// Save result
ml.MeshSave.toAnySupportedFormat(result, 'mesh_offset.stl');
