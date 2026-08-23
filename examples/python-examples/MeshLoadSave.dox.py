import meshlib.mrmeshpy as mrmeshpy
import sys

# This example needs an input mesh; create one so it runs standalone
mrmeshpy.saveMesh(mrmeshpy.makeCube(), "mesh.stl")

# Load mesh
try:
    mesh = mrmeshpy.loadMesh("mesh.stl")
except RuntimeError as e:
    print(f"Failed to load mesh: {e}")
    sys.exit(1)

print(f"Loaded {mesh.topology.numValidVerts()} vertices, "
      f"{mesh.topology.numValidFaces()} faces")

# Save mesh
try:
    mrmeshpy.saveMesh(mesh, "mesh.ply")
except RuntimeError as e:
    print(f"Failed to save mesh: {e}")
    sys.exit(1)

print("Saved mesh.ply")
