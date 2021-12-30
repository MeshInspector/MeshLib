from helper import *

torus = mrmesh.make_outer_half_test_torus(2, 1, 10, 10, None)
#mrmesh.save_mesh(torus, "/home/tim/models/testTorus_half.stl")

torus = mrmesh.make_undercut_test_torus(2, 1, 1.5, 10, 10, None)
#mrmesh.save_mesh(torus, "/home/tim/models/testTorus_undercut.stl")

torus = mrmesh.make_spikes_test_torus(2, 1, 2.5, 10, 10, None)
#mrmesh.save_mesh(torus, "/home/tim/models/testTorus_spikes.stl")

torus = mrmesh.make_components_test_torus(2, 1, 10, 10, None)
#mrmesh.save_mesh(torus, "/home/tim/models/testTorus_components.stl")

torus = mrmesh.make_selfintersect_test_torus(2, 1, 10, 10, None)
#mrmesh.save_mesh(torus, "/home/tim/models/testTorus_selfintersect.stl")
