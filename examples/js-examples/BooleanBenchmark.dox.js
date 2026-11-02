import * as ml from '@meshinspector/meshlib';

await ml.init();

// the number of Booleans can be given in the command line
const numBooleans = process.argv.length > 2 ? Number(process.argv[2]) : 2000;

// two identical spheres of radius 1
using sphereParams = new ml.SphereParams();
sphereParams.radius = 1.0;
sphereParams.numMeshVertices = 3366;
using sphereA = ml.makeSphere(sphereParams);
using sphereB = new ml.Mesh(sphereA);

// subtracts sphereB rotated around Z-axis by i * 0.1 degrees from sphereA;
// sphereB itself is not modified, so the AABB trees of both meshes are built once and reused
function subtractRotated(i) {
  using rigidB2A = ml.AffineXf3f.linear(ml.Matrix3f.rotation(ml.Vector3f.plusZ(), i * 0.1 * Math.PI / 180));
  using params = new ml.BooleanParameters();
  params.rigidB2A = rigidB2A;
  using result = ml.boolean(sphereA, sphereB, ml.BooleanOperation.DifferenceAB, params);
  if (!result.valid())
    throw new Error(result.errorString.toString());
}

// untimed warm-up Boolean, it builds the AABB trees
subtractRotated(1);

const start = performance.now();
for (let i = 1; i <= numBooleans; ++i)
  subtractRotated(i);
const seconds = (performance.now() - start) / 1000;

console.log(`${numBooleans} Booleans in ${seconds.toFixed(3)} s, ${(numBooleans / seconds).toFixed(1)} Booleans/s`);
