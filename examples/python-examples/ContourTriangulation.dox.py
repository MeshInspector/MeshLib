from meshlib import mrmeshpy as mm
from pathlib import Path

wdir = Path(__file__).parent

dm = mm.loadDistanceMapFromImage(wdir / "logo.jpg") # load image
pl2 = mm.distanceMapTo2DIsoPolyline(dm, 210) # make iso lines from image
logo_mesh = mm.PlanarTriangulation.triangulateContours(pl2.contours()) # triangulate isolines

mm.saveMesh(logo_mesh, wdir / "LogoMesh.ctm")
