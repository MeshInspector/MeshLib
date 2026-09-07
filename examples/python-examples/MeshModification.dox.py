import math

import meshlib.mrmeshpy as mrmeshpy

mesh = mrmeshpy.makeTorus()

# Relax mesh (5 iterations)
relax_params = mrmeshpy.MeshRelaxParams()
relax_params.iterations = 5
mrmeshpy.relax(mesh, relax_params)

# Subdivide mesh
props = mrmeshpy.SubdivideSettings()
props.maxDeviationAfterFlip = 0.5
mrmeshpy.subdivideMesh(mesh, props)

# Rotate mesh
rotation_xf = mrmeshpy.AffineXf3f.linear(mrmeshpy.Matrix3f.rotation(mrmeshpy.Vector3f.plusZ(), math.pi * 0.5))
mesh.transform(rotation_xf)
