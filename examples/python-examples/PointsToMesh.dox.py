from meshlib import mrmeshpy as mm
from pathlib import Path

pc = mm.loadPoints("Points.ply")
nefertiti_mesh = mm.triangulatePointCloud(pc)
mm.saveMesh(nefertiti_mesh, "Mesh.ctm")
