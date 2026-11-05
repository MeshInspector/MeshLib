import * as ml from '@meshinspector/meshlib';

await ml.init();

// Load meshes
using meshFloating = ml.MeshLoad.fromAnySupportedFormat('meshA.stl');
using meshFixed = ml.MeshLoad.fromAnySupportedFormat('meshB.stl');

// Prepare ICP parameters
const diagonal = meshFixed.getBoundingBox().diagonal();
const icpSamplingVoxelSize = diagonal * 0.01; // To sample points from object
using icpParams = new ml.ICPProperties();
icpParams.distThresholdSq = (diagonal * 0.1) ** 2; // Use point pairs with maximum distance specified
icpParams.exitVal = diagonal * 0.003; // Stop when this distance reached

// Pair each mesh with its initial transformation (identity here). The wrappers reference the
// meshes, which are declared first so they outlive the wrappers.
using floating = new ml.MeshOrPointsXf(new ml.MeshOrPoints(meshFloating), new ml.AffineXf3f());
using fixed = new ml.MeshOrPointsXf(new ml.MeshOrPoints(meshFixed), new ml.AffineXf3f());

// Calculate transformation
using icp = new ml.ICP(floating, fixed, icpSamplingVoxelSize);
icp.setParams(icpParams);
using xf = icp.calculateTransformation();

// Transform floating mesh
meshFloating.transform(xf);

// Output information string
console.log(icp.getStatusInfo());

// Save result
ml.MeshSave.toAnySupportedFormat(meshFloating, 'meshA_icp.stl');
