import * as ml from '@meshinspector/meshlib';

await ml.init();

// Create some mesh
using mesh = ml.makeTorus(1.0, 0.1, 16, 16);

// Create VertScalars with weights for every vertex: individual extra offset sizes for the points
const positions = mesh.points.toArray(); // x, y, z per vertex
const weights = new Float32Array(positions.length / 3);
let maxWeight = 0;
for (let v = 0; v < weights.length; ++v) {
  weights[v] = Math.abs(positions[3 * v] / 5);
  maxWeight = Math.max(maxWeight, weights[v]);
}
using scalars = ml.VertScalars.fromArray(weights);

// ===params
using params = new ml.WeightedShell.ParametersMetric();
// Algorithm is voxel based, voxel size affects performance and form of result mesh
params.voxelSize = ml.suggestVoxelSize(new ml.MeshPart(mesh), 10000);
// common basic offset applied for all points;
// vertex-specific weighted offsets applied after the basic one
params.offset = 0.2;
params.dist.maxWeight = maxWeight; // should always have maximum between weights provided

// ===offset
using result = ml.WeightedShell.meshShell(mesh, scalars, params);
ml.MeshSave.toAnySupportedFormat(result, 'offset_weighted.ctm');
