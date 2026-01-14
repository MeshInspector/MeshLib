from meshlib import mrmeshpy as mm

def remove_small_components(mesh : mm.Mesh, minAreaRatio : float ):
    mesh.deleteFaces(mesh.topology.getValidFaces() - mm.MeshComponents.getLargeByAreaComponents(mesh,minAreaRatio*mesh.area(),None))

def unite_large_components(mesh : mm.Mesh, minAreaRatio : float )->mm.Mesh:
    remove_small_components(mesh,minAreaRatio) # remove small first
    meshes = []
    allComps = mm.MeshComponents.getAllComponents(mesh)
    meshesPtrs = mm.vectorConstMeshPtr()
    meshesPtrs.resize(len(allComps))
    for i in range(len(allComps)):
        meshes.append(mm.Mesh())
        meshes[i].addMeshPart( mm.MeshPart( mesh,allComps[i] ) )
        meshesPtrs[i] = meshes[i]
    uniteParams = mm.UniteManyMeshesParams()
    uniteParams.fixDegenerations = True
    uniteParams.maxAllowedError = mesh.computeBoundingBox().diagonal()*1e-4
    uniteParams.nestedComponentsMode = mm.NestedComponenetsMode.Merge
    uniteParams.mergeOnFail = True
    return mm.uniteManyMeshes(meshesPtrs,uniteParams)

def find_disorientations(mesh:mm.Mesh):
    dParams = mm.FindDisorientationParams()
    dParams.mode = mm.FindDisorientationParams.RayMode.Shallowest
    dParams.virtualFillHoles = False
    res = mm.findDisorientedFaces(mesh,dParams)
    expParams = mm.MeshComponents.ExpandToComponentsParams()
    expParams.incidence = mm.MeshComponents.FaceIncidence.PerVertex
    expParams.coverRatio = 0.5
    expRes = mm.MeshComponents.expandToComponents(mesh,res,expParams)
    return expRes
     
def fix_disorientations(mesh:mm.Mesh):
    disorientations = find_disorientations(mesh)
    de = mm.getIncidentEdges(mesh.topology,disorientations)
    mesh.topology.flipOrientation(de)
    mesh.invalidateCaches(False)

def close_small_holes(mesh : mm.Mesh, maxParimeter : float):
    mesh.deleteFaces(mm.findHoleComplicatingFaces(mesh))
    holeIds = mesh.topology.findHoleRepresentiveEdges()
    smallHoleIds = []
    for i in range(len(holeIds)):
        if mesh.holePerimeter(holeIds[i]) < maxParimeter:
            smallHoleIds.append(holeIds[i])
    fillSettings = mm.FillHoleNicelySettings()
    fillSettings.triangulateParams.metric = mm.getMinAreaMetric(mesh)
    fillSettings.triangulateParams.multipleEdgesResolveMode = mm.FillHoleParams.MultipleEdgesResolveMode.Strong
    fillSettings.triangulateParams.smoothBd = True
    fillSettings.maxEdgeLen = mesh.averageEdgeLength()
    fillSettings.triangulateOnly = False
    fillSettings.smoothCurvature = False
    fillSettings.naturalSmooth = False
    fillSettings.maxEdgeSplits = 20000
    fillSettings.edgeWeights = mm.EdgeWeights.Cotan
    fillSettings.vmass = mm.VertexMass.NeiArea
    for smallHoleId in smallHoleIds:
        mm.fillHoleNicely(mesh,smallHoleId,fillSettings)

def find_small_tunnel_faces(mesh:mm.Mesh):
    dtSettings = mm.DetectTunnelSettings()
    dtSettings.maxTunnelLength = 1e-2
    return mm.detectTunnelFaces(mesh,dtSettings)

def fix_small_tunnels(mesh:mm.Mesh):
    tunnels = find_small_tunnel_faces(mesh)
    mm.expand(mesh.topology,tunnels)
    mesh.deleteFaces(tunnels)
    newHoles = mesh.topology.findHoleRepresentiveEdges()
    for e in newHoles:
        lf = mesh.topology.left(e)
        rf = mesh.topology.right(e)
        if ( lf.valid() and tunnels.test(lf) ) or ( rf.valid() and tunnels.test(rf) ):
            mm.fillHole(mesh,e)

def fix_self_intersections(mesh:mm.Mesh):
    params = mm.SelfIntersections.Settings()
    params.maxExpand = 2
    params.relaxIterations = 2
    params.method = mm.SelfIntersections.Settings.Method.CutAndFill
    params.touchIsIntersection = True
    mm.SelfIntersections.fix(mesh,params)

def fix_degeneracies( mesh : mm.Mesh, tolerance : float, force : bool ):
    mm.fixMultipleEdges(mesh)
    params = mm.FixMeshDegeneraciesParams()
    params.maxDeviation = 0.1*tolerance
    params.tinyEdgeLength = 0.01*tolerance
    params.criticalTriAspectRatio = 1e3
    params.mode = mm.FixMeshDegeneraciesParams.Mode.Remesh
    if force:
        params.mode = mm.FixMeshDegeneraciesParams.Mode.RemeshPatch
    mm.fixMeshDegeneracies(mesh,params)

def local_repair(mesh : mm.Mesh, tolerance : float, force : bool)->mm.Mesh:
    mm.MeshBuilder.uniteCloseVertices(mesh,0.0,True)
    mm.duplicateMultiHoleVertices(mesh)
    remove_small_components(mesh,0.002)
    fix_disorientations(mesh)
    close_small_holes(mesh,mesh.computeBoundingBox().diagonal()/3.0)
    fix_small_tunnels(mesh)
    fix_self_intersections(mesh)
    mesh = unite_large_components(mesh,0.002)
    fix_degeneracies(mesh,tolerance,force)
    return mesh

def has_issues(mesh : mm.Mesh,minAreaRatio,maxHolePerimeter,tolerance):
    if mm.MeshComponents.getNumComponents(mesh) > 1:
        return True
    holeIds = mesh.topology.findHoleRepresentiveEdges()
    for i in range(len(holeIds)):
        if mesh.holePerimeter(holeIds[i]) < maxHolePerimeter:
            return True
    selfies = mm.SelfIntersections.getFaces(mesh)
    if selfies.count() > 0:
        return True
    multipleEdges = mm.findMultipleEdges(mesh.topology)
    if multipleEdges.size() > 0:
        return True
    degenFaces = mm.findDegenerateFaces(mesh,1e3)
    if degenFaces.count() > 0:
        return True 
    shortEdges = mm.findShortEdges(mesh,0.01*tolerance)
    if shortEdges.count() > 0:
        return True
    tunnels = find_small_tunnel_faces(mesh)
    if tunnels.count() > 0:
        return True
    disorientations = find_disorientations(mesh)
    if disorientations.count() > 0:
        return True
    return False

def has_overhangs( mesh : mm.Mesh, layerStep : float, width : float )->bool:
	oParams = mm.FindOverhangsSettings()
	oParams.layerHeight = layerStep
	oParams.maxOverhangDistance = width
	return mm.findOverhangs(mesh,oParams).size() > 0

def has_big_holes(mesh : mm.Mesh, critHoleLength : float)->bool:
	if ( critHoleLength < 0 ):
		return False
	holeIds = mesh.topology.findHoleRepresentiveEdges()
	for holeId in holeIds:
		if ( mesh.holePerimeter( holeId ) >= critHoleLength ):
			return True
	return False

# one can use default tolerance as "mesh.computeBoundingBox().diagonal()*2.5e-3"
def iterative_repair(mesh : mm.Mesh, tolerance: float):
    diagonal = mesh.computeBoundingBox().diagonal()
    for i in range(3):
        if not has_issues(mesh,0.002,diagonal/3.0,tolerance):
            break
        mesh = local_repair(mesh,tolerance,i > 1) # last iteration is forced
    return mesh