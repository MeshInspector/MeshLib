from meshlib import mrmeshpy as mm
from pathlib import Path

colors = mm.VertColors()
lps = mm.PointsLoadSettings()
lps.colors = colors
pc = mm.loadPoints("TerrainPoints.ply",lps)

terrain_mesh = mm.terrainTriangulation(pc.points.vec)

mss = mm.SaveSettings()
mss.colors = colors
mm.saveMesh(terrain_mesh, "TerrainMesh.ctm",mss)
