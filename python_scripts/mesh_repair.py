from meshlib import mrmeshpy as mm

def remove_small_components(mesh : mm.Mesh, minAreaRatio : float ):
    mesh.deleteFaces(mesh.topology.getValidFaces() - mm.MeshComponents.getLargeByAreaComponents(mesh,minAreaRatio*mesh.area(),None))

def unite_large_components(mesh : mm.Mesh, minAreaRatio : float ):
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
    # uniteParams.mergeOnFail = True # this will appear in next release
    mesh = mm.uniteManyMeshes(meshesPtrs,uniteParams)

def close_small_holes(mesh : mm.Mesh, maxParimeter : float):
    mesh.deleteFaces(mm.findHoleComplicatingFaces(mesh))
    holeIds = mesh.topology.findHoleRepresentiveEdges()
    smallHoleIds = []
    for i in range(len(holeIds)):
        if mesh.holePerimiter(holeIds[i]) < maxParimeter:
            smallHoleIds.append(holeIds[i])
    fillSettings = mm.FillHoleNicelySettings()
    fillSettings.triangulateParams.metric = mm.getMinAreaMetric(mesh)
    fillSettings.triangulateParams.multipleEdgesResolveMode = mm.FillHoleParams.MultipleEdgesResolveMode.Strong
    fillSettings.maxEdgeLen = mesh.averageEdgeLength()
    fillSettings.triangulateOnly = False
    fillSettings.smoothCurvature = False
    fillSettings.maxEdgeSplits = 20000
    for smallHoleId in smallHoleIds:
        mm.fillHoleNicely(mesh,smallHoleId,fillSettings)

def fix_self_intersections(mesh:mm.Mesh):
    params = mm.SelfIntersections.Settings()
    params.maxExpand = 2
    params.relaxIterations = 2
    params.method = mm.SelfIntersections.Settings.Method.CutAndFill
    params.touchIsIntersection = False # required to disable subdivision on fixing self-intersections
    print(params.subdivideEdgeLen)
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

def local_repair(mesh : mm.Mesh, tolerance : float, force : bool):
    mm.MeshBuilder.uniteCloseVertices(mesh,0.0)
    remove_small_components(mesh,0.002)
    close_small_holes(mesh,mesh.computeBoundingBox().diagonal()/3.0)
    fix_self_intersections(mesh)
    unite_large_components(mesh,0.002)
    fix_degeneracies(mesh,tolerance,force)

def has_issues(mesh : mm.Mesh,minAreaRatio,maxHolePerimeter,tolerance):
    smallComps = mesh.topology.getValidFaces() - mm.MeshComponents.getLargeByAreaComponents(mesh,minAreaRatio*mesh.area(),None)
    if smallComps.count() > 0:
        return True
    holeIds = mesh.topology.findHoleRepresentiveEdges()
    for i in range(len(holeIds)):
        if mesh.holePerimiter(holeIds[i]) < maxHolePerimeter:
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
		if ( mesh.holePerimiter( holeId ) >= critHoleLength ):
			return True
	return False

def iterative_repair(mesh : mm.Mesh):
    diagonal = mesh.computeBoundingBox().diagonal()
    tolerance = 2.5*diagonal*1e-3
    for i in range(4): # one more iteration then in MeshInspector
        if not has_issues(mesh,0.002,diagonal/3.0,tolerance):
            break
        local_repair(mesh,tolerance,i > 2) # last iteration is forced
    return mesh
