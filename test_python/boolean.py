from helper import *

torusIntersected = mrmesh.make_selfintersect_test_torus(2, 1, 10, 10, None)
mrmesh.fix_self_intersections(torusIntersected, 0.1)

torus = mrmesh.make_torus(2, 1, 10, 10, None)

transVector = mrmesh.Vector3()
transVector.x=0.5
transVector.y=1
transVector.z=1

diffXf = mrmesh.AffineXf3.translation(transVector)

torus2 = mrmesh.make_torus(2, 1, 10, 10, None)
torus2.transform(diffXf)

torus1 = torus
p = torus1.points.vec.size()
mrmesh.boolean_sub(torus1, torus2, 0.05)
p_sub = torus1.points.vec.size()

torus1 = torus
mrmesh.boolean_union(torus1, torus2, 0.05)
p_union = torus1.points.vec.size()

torus1 = torus
mrmesh.boolean_intersect(torus1, torus2, 0.05)
p_intersect = torus1.points.vec.size()

assert( p == 100)

import math
assert( math.isclose( p_sub, p_intersect, rel_tol=1e-2) )
assert( math.isclose( p_sub / p, 400, abs_tol=100) )
assert( math.isclose( p_sub / p_union, 0.7, rel_tol=0.2) )
