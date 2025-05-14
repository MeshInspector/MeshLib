from meshlib import mrmeshpy as mm
from pathlib import Path

wdir = Path(__file__).parent
pc = mm.loadPoints(wdir / "Points.ply")
nefertiti_mesh = mm.triangulatePointCloud(pc)
mm.saveMesh(nefertiti_mesh, wdir / "Mesh.ctm")
