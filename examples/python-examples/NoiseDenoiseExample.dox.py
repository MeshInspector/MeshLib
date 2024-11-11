from meshlib import mrmeshpy as mm
 
# Load mesh
mesh = mm.loadMesh("mesh.stl")
 
# Create parameters for adding noise
nSettings = mm.NoiseSettings()
nSettings.sigma = mesh.computeBoundingBox().diagonal()*0.0001

# Add noise to mesh
mm.addNoise(mesh.points,mesh.topology.getValidVerts(),nSettings)

# Save noised mesh
mm.saveMesh(mesh,"noised_mesh.stl")

# Denoise mesh with sharpening for sharp edges
# see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
mm.meshDenoiseViaNormals( mesh )

# Save denoised mesh
mm.saveMesh(mesh,"denoised_mesh.stl")
