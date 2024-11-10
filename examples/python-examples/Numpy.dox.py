import meshlib.mrmeshnumpy as mrmeshnumpy
import numpy as np

faces = np.ndarray(
    shape=(2, 3),
    dtype=np.int32,
    buffer=np.array(
        [
            [0, 1, 2],
            [2, 3, 0],
        ],
        dtype=np.int32,
    ),
)

# mrmesh uses float32 for vertex coordinates
# however, you could also use float64
verts = np.ndarray(
    shape=(4, 3),
    dtype=np.float32,
    buffer=np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ],
        dtype=np.float32,
    ),
)

mesh = mrmeshnumpy.meshFromFacesVerts(faces, verts)

# some mesh manipulations

out_verts = mrmeshnumpy.getNumpyVerts(mesh)
out_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)
