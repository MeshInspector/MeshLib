### windows-only: [

# Fixes DLL loading paths.

def _init_patch():
    import os
    libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    os.add_dll_directory(libs_dir)

_init_patch()
del _init_patch

### ]



# Manually define some aliases. It's recommended to avoid those.

def _init_patch():
    from . import mrmeshpy
    mrmeshpy.BooleanResMapObj = mrmeshpy.BooleanResultMapper_MapObject
    mrmeshpy.copyMesh = mrmeshpy.Mesh
    mrmeshpy.FaceMap.vec = mrmeshpy.FaceMap.vec_
    mrmeshpy.FaceNormals.vec = mrmeshpy.FaceNormals.vec_
    mrmeshpy.findUndercuts = mrmeshpy.FixUndercuts_findUndercuts
    mrmeshpy.FixSelfIntersectionMethod = mrmeshpy.SelfIntersections_Settings_Method
    mrmeshpy.FixSelfIntersectionSettings = mrmeshpy.SelfIntersections_Settings
    mrmeshpy.fixUndercuts = mrmeshpy.FixUndercuts_fixUndercuts
    mrmeshpy.GeneralOffsetParametersMode = mrmeshpy.GeneralOffsetParameters_Mode
    mrmeshpy.getAllComponents = mrmeshpy.MeshComponents_getAllComponents
    mrmeshpy.getAllComponentsVerts = mrmeshpy.MeshComponents_getAllComponentsVerts
    mrmeshpy.ICP.getLastICPInfo = mrmeshpy.ICP.getStatusInfo
    mrmeshpy.LaplacianEdgeWeightsParam = mrmeshpy.EdgeWeights
    mrmeshpy.loadLines = mrmeshpy.LinesLoad_fromAnySupportedFormat
    mrmeshpy.loadMesh = mrmeshpy.MeshLoad_fromAnySupportedFormat
    mrmeshpy.loadPoints = mrmeshpy.PointsLoad_fromAnySupportedFormat
    mrmeshpy.loadVoxelsGav = mrmeshpy.VoxelsLoad_fromGav
    mrmeshpy.loadVoxelsRaw = mrmeshpy.VoxelsLoad_fromRaw
    mrmeshpy.localFindSelfIntersections = mrmeshpy.SelfIntersections_getFaces
    mrmeshpy.localFixSelfIntersections = mrmeshpy.SelfIntersections_fix
    mrmeshpy.MeshBuilderSettings = mrmeshpy.MeshBuilder_BuildSettings
    mrmeshpy.MeshToVolumeParamsType = mrmeshpy.MeshToVolumeParams_Type
    mrmeshpy.ObjectDistanceMap.extractDistanceMap = mrmeshpy.ObjectDistanceMap.getDistanceMap
    mrmeshpy.ObjectLines.extractLines = mrmeshpy.ObjectLines.polyline
    mrmeshpy.ObjectMesh.extractMesh = mrmeshpy.ObjectMesh.mesh
    mrmeshpy.ObjectPoints.extractPoints = mrmeshpy.ObjectPoints.pointCloud
    mrmeshpy.objectSave = mrmeshpy.ObjectSave_toAnySupportedFormat
    mrmeshpy.ObjectVoxels.extractVoxels = mrmeshpy.ObjectVoxels.vdbVolume
    mrmeshpy.saveAllSlicesToImage = mrmeshpy.VoxelsSave_saveAllSlicesToImage
    mrmeshpy.saveMesh = mrmeshpy.MeshSave_toAnySupportedFormat
    mrmeshpy.savePoints = mrmeshpy.PointsSave_toAnySupportedFormat
    mrmeshpy.saveSliceToImage = mrmeshpy.VoxelsSave_saveSliceToImage
    mrmeshpy.saveVoxelsGav = mrmeshpy.VoxelsSave_toGav
    mrmeshpy.saveVoxelsRaw = mrmeshpy.VoxelsSave_toRawAutoname
    mrmeshpy.TextAlignParams = mrmeshpy.TextMeshAlignParams
    mrmeshpy.topologyFromTriangles = mrmeshpy.MeshBuilder_fromTriangles
    mrmeshpy.triangulateContours = mrmeshpy.PlanarTriangulation_triangulateContours
    mrmeshpy.Triangulation.vec = mrmeshpy.Triangulation.vec_
    mrmeshpy.uniteCloseVertices = mrmeshpy.MeshBuilder_uniteCloseVertices
    mrmeshpy.vectorConstMeshPtr = mrmeshpy.std_vector_const_Mesh
    mrmeshpy.vectorEdges = mrmeshpy.EdgeLoop
    mrmeshpy.VertCoords.vec = mrmeshpy.VertCoords.vec_
    mrmeshpy.VertScalars.vec = mrmeshpy.VertScalars.vec_
    mrmeshpy.VoxelsSaveSavingSettings = mrmeshpy.VoxelsSave_SavingSettings

_init_patch()
del _init_patch
