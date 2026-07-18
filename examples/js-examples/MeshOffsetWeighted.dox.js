import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

using mesh = ml.makeTorus(1.0, 0.1, 16, 16);

// build a per-vertex weight (extra offset) from each vertex's x coordinate
using coords = mesh.points;
const positions = coords.toArray();
const weights = new Float32Array(positions.length / 3);
let maxWeight = 0;
for (let v = 0; v < weights.length; ++v) {
  weights[v] = Math.abs(positions[3 * v] / 5);
  maxWeight = Math.max(maxWeight, weights[v]);
}
using scalars = ml.VertScalars.fromArray(weights);

using params = new ml.WeightedShellParametersMetric();
// the algorithm is voxel-based; the voxel size affects performance and the result shape
params.voxelSize = ml.suggestVoxelSize(mesh, 1000);
// a common basic offset applied to every vertex; the weighted offsets are added on top
params.offset = 0.2;
// the maximum weight must be an upper bound of the provided weights
using dist = params.dist;
dist.maxWeight = maxWeight;
params.dist = dist;

using result = ml.WeightedShell.meshShell(mesh, scalars, params);

ml.MeshSave.toAnySupportedFormat(result, 'offset_weighted.ply');
