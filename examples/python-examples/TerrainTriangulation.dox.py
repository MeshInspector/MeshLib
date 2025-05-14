from meshlib import mrmeshpy as mm
from pathlib import Path

wdir = Path(__file__).parent

colors = mm.VertColors()
lps = mm.PointsLoadSettings()
lps.colors = colors
pc = mm.loadPoints(wdir / "TerrainScan.e57",lps)

terrain_mesh = mm.terrainTriangulation(pc.points.vec)

mss = mm.SaveSettings()
mss.colors = colors
mm.saveMesh(terrain_mesh, wdir / "TerrainMesh.ctm",mss)
