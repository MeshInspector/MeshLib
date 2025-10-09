from meshlib import mrmeshpy as mm
from meshlib import mrcudapy as mc

def hasBigHoles(mesh : mm.Mesh, critHoleLength : float)->bool:
	if ( critHoleLength < 0 ):
		return False
	holeIds = mesh.topology.findHoleRepresentiveEdges()
	for holeId in holeIds:
		if ( mesh.holePerimiter( holeId ) >= critHoleLength ):
			return True
	return False

def hasOverhangs( mesh : mm.Mesh, layerStep : float, width : float )->bool:
	oParams = mm.FindOverhangsSettings()
	oParams.layerHeight = layerStep
	oParams.maxOverhangDistance = width
	return mm.findOverhangs(mesh,oParams).size() > 0

def heal( mesh : mm.Mesh, voxelSize : float, decimate : bool = True )->mm.Mesh:
	params = mm.RebuildMeshSettings()
	params.voxelSize = voxelSize # mm.suggestVoxelSize(mesh,5e7) # suggest amount of voxels to use
	if mc.isCudaAvailable():
		params.fwn = mc.FastWindingNumber(mesh) # only if cuda is available, for speedup
	params.decimate = decimate
	return mm.rebuildMesh(mesh,params) # it can leave some small components
