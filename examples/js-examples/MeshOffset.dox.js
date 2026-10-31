import * as ml from '@meshinspector/meshlib';

await ml.init();

// Load mesh
using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// Setup parameters
using params = new ml.OffsetParameters();
params.voxelSize = mesh.computeBoundingBox().diagonal() * 5e-3; // offset grid precision (algorithm is voxel based)
using boundaries = ml.findRightBoundary(mesh.topology);
if (!boundaries.empty())
  params.signDetectionMode = ml.SignDetectionMode.HoleWindingRule; // use if you have holes in mesh

// Make offset mesh
const offset = mesh.computeBoundingBox().diagonal() * 0.05;
let resultMesh;
try {
  resultMesh = ml.offsetMesh(new ml.MeshPart(mesh), offset, params);
} catch (e) {
  console.log(e.message);
  process.exit(1);
}

// Save result
ml.MeshSave.toAnySupportedFormat(resultMesh, 'offsetMesh.stl');
resultMesh.delete();
