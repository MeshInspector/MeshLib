import * as ml from '@meshinspector/meshlib';

await ml.init();

// generate a point cloud sampling a unit sphere
using pointCloud = new ml.PointCloud();
const resolution = 100;
for (let i = 0; i < resolution; ++i) {
  const u = 2 * Math.PI * i / (resolution - 1);
  for (let j = 0; j < resolution; ++j) {
    const v = Math.PI * j / (resolution - 1);
    pointCloud.addPoint(new ml.Vector3f(Math.cos(u) * Math.sin(v), Math.sin(u) * Math.sin(v), Math.cos(v)));
  }
}

// remove the points that are too close (e.g. the duplicated poles)
using samplingSettings = new ml.UniformSamplingSettings();
samplingSettings.distance = 1e-3;
using samples = ml.pointUniformSampling(pointCloud, samplingSettings);
const sampled = samples.value();
if (!sampled)
  throw new Error('sampling was cancelled');
pointCloud.validPoints = sampled;
pointCloud.invalidateCaches();

// triangulate the sampled cloud
using triangulated = ml.triangulatePointCloud(pointCloud);
const triangulatedMesh = triangulated.value();
if (!triangulatedMesh)
  throw new Error('triangulation was cancelled');

// fix possible issues with a zero offset
using offsetParams = new ml.OffsetParameters();
using triangulatedPart = new ml.MeshPart(triangulatedMesh);
offsetParams.voxelSize = ml.suggestVoxelSize(triangulatedPart, 5e6);
using mesh = ml.offsetMesh(triangulatedPart, 0, offsetParams);

ml.MeshSave.toAnySupportedFormat(mesh, 'result.stl');
