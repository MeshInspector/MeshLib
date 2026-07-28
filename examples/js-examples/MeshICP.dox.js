import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// Load meshes
using meshFloating = ml.MeshLoad.fromAnySupportedFormat('meshA.stl');
using meshFixed = ml.MeshLoad.fromAnySupportedFormat('meshB.stl');

// Prepare ICP parameters
using box = meshFixed.computeBoundingBox();
const diagonal = box.diagonal();
const icpSamplingVoxelSize = diagonal * 0.01; // To sample points from object
using icpParams = new ml.ICPProperties();
icpParams.distThresholdSq = (diagonal * 0.1) ** 2; // Use point pairs with maximum distance specified
icpParams.exitVal = diagonal * 0.003; // Stop when distance reached

// Pair each mesh with an initial transformation (identity here). The JS ICP takes
// MeshOrPointsXf; the source meshes back these wrappers, and the reverse-order disposal
// of `using` guarantees the meshes outlive them.
using floatingMop = ml.MeshOrPoints.fromMesh(meshFloating);
using fixedMop = ml.MeshOrPoints.fromMesh(meshFixed);
using identityXf = new ml.AffineXf3f();
using floating = new ml.MeshOrPointsXf(floatingMop, identityXf);
using fixed = new ml.MeshOrPointsXf(fixedMop, identityXf);

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
