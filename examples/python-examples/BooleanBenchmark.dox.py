import math
import sys
import time

import meshlib.mrmeshpy as mrmeshpy

# the number of Booleans can be given in the command line
num_booleans = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

# two identical spheres of radius 1
sphere_params = mrmeshpy.SphereParams()
sphere_params.radius = 1.0
sphere_params.numMeshVertices = 3366
sphere_a = mrmeshpy.makeSphere(sphere_params)
sphere_b = mrmeshpy.copyMesh(sphere_a)


# subtracts sphere_b rotated around Z-axis by i * 0.1 degrees from sphere_a;
# sphere_b itself is not modified, so the AABB trees of both meshes are built once and reused
def subtract_rotated(i):
    rigid_b2a = mrmeshpy.AffineXf3f.linear(mrmeshpy.Matrix3f.rotation(mrmeshpy.Vector3f.plusZ(), math.radians(i * 0.1)))
    result = mrmeshpy.boolean(sphere_a, sphere_b, mrmeshpy.BooleanOperation.DifferenceAB, rigid_b2a)
    if not result.valid():
        raise RuntimeError(result.errorString)


# untimed warm-up Boolean, it builds the AABB trees
subtract_rotated(1)

start = time.perf_counter()
for i in range(1, num_booleans + 1):
    subtract_rotated(i)
seconds = time.perf_counter() - start

print(f"{num_booleans} Booleans in {seconds:.3f} s, {num_booleans / seconds:.1f} Booleans/s")
