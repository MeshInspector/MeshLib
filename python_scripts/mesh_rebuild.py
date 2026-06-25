from meshlib import mrmeshpy as mm
from meshlib import mrcudapy as mc

def rebuild( mesh : mm.Mesh, voxelSize : float, decimate : bool = True )->mm.Mesh:
	params = mm.RebuildMeshSettings()
	params.voxelSize = voxelSize # mm.suggestVoxelSize(mesh,5e7) # suggest amount of voxels to use
	if mc.isCudaAvailable():
		params.fwn = mc.FastWindingNumber(mesh) # only if cuda is available, for speedup
	params.decimate = decimate
	return mm.rebuildMesh(mesh,params) # it can leave some small components
