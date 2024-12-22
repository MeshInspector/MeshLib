from meshlib import mrmeshpy as mlib
from meshlib import mrcudapy as mc


mesh = mlib.loadMesh("mesh.stl")
offset_params = mlib.GeneralOffsetParameters()
offset_params.signDetectionMode = mlib.SignDetectionMode.HoleWindingRule
offset_params.fwn = mc.FastWindingNumber(mesh) # Enables usage of CUDA
offset_params.voxelSize = mlib.suggestVoxelSize(mesh, 5e6)

new_mesh = mlib.generalOffsetMesh(mp=mesh, offset=0, params=offset_params)
mlib.saveMesh(new_mesh, f"cuda_offset.ctm")
