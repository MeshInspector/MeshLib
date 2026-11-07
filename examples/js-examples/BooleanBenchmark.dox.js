import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// the number of Booleans can be given in the command line
const numBooleans = process.argv.length > 2 ? Number(process.argv[2]) : 2000;

// two identical spheres of radius 1
using sphereA = ml.makeSphere({ radius: 1.0, numMeshVertices: 3366 });
using sphereB = ml.makeSphere({ radius: 1.0, numMeshVertices: 3366 });

// subtracts sphereB rotated around Z-axis by i * 0.1 degrees from sphereA;
// sphereB itself is not modified, so the AABB trees of both meshes are built once and reused
function subtractRotated(i) {
  using rotation = ml.Matrix3f.rotation(ml.Vector3.plusZ(), i * 0.1 * Math.PI / 180);
  using rigidB2A = ml.AffineXf3f.linear(rotation);
  using result = ml.booleanWithXf(sphereA, sphereB, ml.BooleanOperation.DifferenceAB, rigidB2A);
  if (!result.valid())
    throw new Error(result.errorString);
}

// untimed warm-up Boolean, it builds the AABB trees
subtractRotated(1);

const start = performance.now();
for (let i = 1; i <= numBooleans; ++i)
  subtractRotated(i);
const seconds = (performance.now() - start) / 1000;

console.log(`${numBooleans} Booleans in ${seconds.toFixed(3)} s, ${(numBooleans / seconds).toFixed(1)} Booleans/s`);
