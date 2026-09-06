from meshlib import mrmeshpy as mm
from pathlib import Path

colors = mm.VertColors()
lps = mm.PointsLoadSettings()
lps.colors = colors
pc = mm.loadPoints("TerrainPoints.ply",lps)

terrain_mesh = mm.delaunayTriangulationXY(pc)

mss = mm.SaveSettings()
if (pc.points.vec.size() == colors.size()):
    mss.colors = colors
mm.saveMesh(terrain_mesh, "TerrainMesh.ctm",mss)
