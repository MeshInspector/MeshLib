import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// the global registration can be applied to meshes and point clouds;
// to keep the sample simple, we work with point clouds only
const inputPaths = ['cloud1.ply', 'cloud2.ply', 'cloud3.ply'];
const outputPath = 'registered.ply';

// a DisposableStack frees the per-input WASM handles (there is a variable number of them) at the
// end of the scope, in reverse order, the same way `using` frees a single handle
using stack = new DisposableStack();

// as ICP and MultiwayICP accept both meshes and point clouds, each input is wrapped in a
// MeshOrPointsXf; the wrapper references the source cloud, so the clouds must outlive it (the
// stack, declared first, is disposed last)
using identityXf = new ml.AffineXf3f();
const inputs = [];
const objects = [];
let maxVolume = -Infinity;
let baseDiagonal = 0;
for (const path of inputPaths) {
  const cloud = stack.use(ml.PointsLoad.fromAnySupportedFormat(path));
  inputs.push(cloud);

  using bbox = cloud.computeBoundingBox();
  const volume = bbox.volume();
  if (volume > maxVolume) {
    maxVolume = volume;
    baseDiagonal = bbox.diagonal();
  }

  const mop = stack.use(ml.MeshOrPoints.fromPoints(cloud));
  objects.push(stack.use(new ml.MeshOrPointsXf(mop, identityXf)));
}

// set the sampling voxel size relative to the largest input
using samplingParams = new ml.MultiwayICPSamplingParameters();
samplingParams.samplingVoxelSize = baseDiagonal * 0.03;

using icp = new ml.MultiwayICP(objects, samplingParams);
using icpProps = new ml.ICPProperties();
icp.setParams(icpProps);

// gather statistics
icp.updateAllPointPairs();
console.log(`Samples: ${icp.getNumSamples()}`);
console.log(`Active point pairs: ${icp.getNumActivePairs()}`);

// one transformation per input object
const xfs = icp.calculateTransformations();

// apply each transformation to its cloud and merge everything into one output cloud
using output = new ml.PointCloud();
for (let i = 0; i < inputs.length; ++i) {
  using xf = xfs[i];
  using srcPoints = inputs[i].points;
  const coords = srcPoints.toArray();
  for (let p = 0; p < coords.length; p += 3)
    output.addPoint(xf.apply({ x: coords[p], y: coords[p + 1], z: coords[p + 2] }));
}

ml.PointsSave.toAnySupportedFormat(output, outputPath);
