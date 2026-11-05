import * as ml from '@meshinspector/meshlib';

await ml.init();

// the global registration can be applied to meshes and point clouds;
// to keep the sample simple, we work with point clouds only
const inputPaths = ['cloud1.ply', 'cloud2.ply', 'cloud3.ply'];
const outputPath = 'registered.ply';

// a DisposableStack frees the per-input handles (there is a variable number of them) at the
// end of the scope, in reverse order, the same way `using` frees a single handle
using stack = new DisposableStack();

// as ICP and MultiwayICP accept both meshes and point clouds, each input is wrapped in a
// MeshOrPointsXf (the object plus its initial transformation); the wrapper references the
// source cloud, so the clouds must outlive it (the stack, declared first, is disposed last)
const inputs = [];
using objects = new ml.Vector_MR_MeshOrPointsXf_MR_ObjId(); // ICPObjects
let maxVolume = -Infinity;
let baseDiagonal = 0;
for (const path of inputPaths) {
  const cloud = stack.use(ml.PointsLoad.fromAnySupportedFormat(path));
  inputs.push(cloud);
  const bbox = cloud.computeBoundingBox();
  if (bbox.volume() > maxVolume) {
    maxVolume = bbox.volume();
    baseDiagonal = bbox.diagonal();
  }
  objects.push_back(new ml.MeshOrPointsXf(new ml.MeshOrPoints(cloud), new ml.AffineXf3f()));
}

// you can set various parameters for the global registration; see the documentation for more info
using samplingParams = new ml.MultiwayICPSamplingParameters();
// set the sampling voxel size relative to the largest input
samplingParams.samplingVoxelSize = baseDiagonal * 0.03;

using icp = new ml.MultiwayICP(objects, samplingParams);
icp.setParams(new ml.ICPProperties());

// gather statistics
icp.updateAllPointPairs();
console.log(`Samples: ${icp.getNumSamples()}`);
console.log(`Active point pairs: ${icp.getNumActivePairs()}`);

// one transformation per input object
console.log('Calculating transformations...');
using xfs = icp.calculateTransformations();

// apply each transformation to its cloud and merge everything into one output cloud
using output = new ml.PointCloud();
for (let i = 0; i < inputs.length; ++i) {
  const xf = xfs.index(new ml.ObjId(i));
  const coords = inputs[i].points.toArray(); // x, y, z per point
  for (let p = 0; p < coords.length; p += 3)
    output.addPoint(xf.call(new ml.Vector3f(coords[p], coords[p + 1], coords[p + 2])));
}
ml.PointsSave.toAnySupportedFormat(output, outputPath);
