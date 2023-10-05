from meshlib import mrmeshpy as mm

def holeLength( mesh : mm.Mesh, hole : mm.vectorEdges )->float:
	sumLength = 0.0
	for e in hole:
		org = mesh.topology.org( e )
		dest = mesh.topology.dest( e )
		sumLength += (mesh.points.vec[dest.get()] - mesh.points.vec[org.get()]).length()
	return sumLength

def hasBigHoles(mesh : mm.Mesh, critHoleLength : float)->bool:
	if ( critHoleLength < 0 ):
		return False
	holes = mm.findRightBoundary( mesh.topology )
	for hole in holes:
		if ( holeLength(mesh, hole) >= critHoleLength ):
			return True
	return False

def hasOverhangs( mesh : mm.Mesh, layerStep : float, width : float )->bool:
	oParams = mm.FindOverhangsSettings()
	oParams.layerHeight = layerStep
	oParams.maxOverhangDistance = width
	return mrmesh.findOverhangs(mesh,oParams).size() > 0

def heal( mesh : mm.Mesh, voxelSize : float )->mm.Mesh:
	oParams = mm.OffsetParameters()
	oParams.voxelSize = voxelSize
	return mm.offsetMesh( mesh, 0.0, oParams)
