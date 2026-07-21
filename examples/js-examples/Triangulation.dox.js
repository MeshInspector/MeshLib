import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// generate a point cloud sampling a unit sphere
using pointCloud = new ml.PointCloud();
const resolution = 100;
for (let i = 0; i < resolution; ++i) {
  const u = 2 * Math.PI * i / (resolution - 1);
  for (let j = 0; j < resolution; ++j) {
    const v = Math.PI * j / (resolution - 1);
    pointCloud.addPoint({
      x: Math.cos(u) * Math.sin(v),
      y: Math.sin(u) * Math.sin(v),
      z: Math.cos(v),
    });
  }
}

// remove the points that are too close (e.g. the duplicated poles)
using samplingSettings = new ml.UniformSamplingSettings();
samplingSettings.distance = 1e-3;
using samples = ml.pointUniformSampling(pointCloud, samplingSettings);
pointCloud.validPoints = samples;
pointCloud.invalidateCaches();

// triangulate the sampled cloud
using triangulationParams = new ml.TriangulationParameters();
using triangulated = ml.triangulatePointCloud(pointCloud, triangulationParams);

// fix possible issues with a zero offset
using offsetParams = new ml.OffsetParameters();
offsetParams.voxelSize = ml.suggestVoxelSize(triangulated, 5e6);
using mesh = ml.offsetMesh(triangulated, 0, offsetParams);

ml.MeshSave.toAnySupportedFormat(mesh, 'result.stl');
