from meshlib import mrmeshpy as mm
from meshlib import mrmeshnumpy as mn
import numpy as np
import plotly.graph_objects as go

# load mesh
mesh = mm.loadMesh("mesh.stl")
# extract numpy arrays
verts = mn.getNumpyVerts(mesh)
faces = mn.getNumpyFaces(mesh.topology)

# prepare data for plotly
verts_t = np.transpose(verts)
faces_t = np.transpose(faces)

# draw
fig = go.Figure(data=[
    go.Mesh3d(
        x=verts_t[0],
        y=verts_t[1],
        z=verts_t[2],
        i=faces_t[0],
        j=faces_t[1],
        k=faces_t[2]
    )
])

fig.show()
