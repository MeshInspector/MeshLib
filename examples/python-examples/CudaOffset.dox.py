from meshlib import mrmeshpy as mlib
from meshlib import mrcudapy as mc


mesh = mlib.loadMesh("mesh.stl")
offset_params = mlib.GeneralOffsetParameters()
offset_params.signDetectionMode = mlib.SignDetectionMode.WindingRule
offset_params.fwn = mc.FastWindingNumber(mesh)
offset_params.voxelSize = mlib.suggestVoxelSize(mesh, 5e6)

new_mesh = mlib.generalOffsetMesh(mp=mesh, offset=1, params=offset_params)
mlib.saveMesh(new_mesh, f"cuda_offset.ctm")
