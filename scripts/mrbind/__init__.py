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
    mrmeshpy.BooleanResMapObj = mrmeshpy.BooleanResultMapper.MapObject
    mrmeshpy.copyMesh = mrmeshpy.Mesh
    mrmeshpy.FaceMap.vec = mrmeshpy.FaceMap.vec_
    mrmeshpy.FaceNormals.vec = mrmeshpy.FaceNormals.vec_
    mrmeshpy.findUndercuts = mrmeshpy.FixUndercuts.findUndercuts
    mrmeshpy.FixSelfIntersectionMethod = mrmeshpy.SelfIntersections.Settings.Method
    mrmeshpy.FixSelfIntersectionSettings = mrmeshpy.SelfIntersections.Settings
    mrmeshpy.fixUndercuts = mrmeshpy.FixUndercuts.fixUndercuts
    mrmeshpy.GeneralOffsetParametersMode = mrmeshpy.GeneralOffsetParameters.Mode
    mrmeshpy.getAllComponents = mrmeshpy.MeshComponents.getAllComponents
    mrmeshpy.getAllComponentsVerts = mrmeshpy.MeshComponents.getAllComponentsVerts
    mrmeshpy.ICP.getLastICPInfo = mrmeshpy.ICP.getStatusInfo
    mrmeshpy.LaplacianEdgeWeightsParam = mrmeshpy.EdgeWeights
    mrmeshpy.loadLines = mrmeshpy.LinesLoad.fromAnySupportedFormat
    mrmeshpy.loadMesh = mrmeshpy.MeshLoad.fromAnySupportedFormat
    mrmeshpy.loadPoints = mrmeshpy.PointsLoad.fromAnySupportedFormat
    mrmeshpy.loadVoxelsGav = mrmeshpy.VoxelsLoad.fromGav
    mrmeshpy.loadVoxelsRaw = mrmeshpy.VoxelsLoad.fromRaw
    mrmeshpy.localFindSelfIntersections = mrmeshpy.SelfIntersections.getFaces
    mrmeshpy.localFixSelfIntersections = mrmeshpy.SelfIntersections.fix
    mrmeshpy.MeshBuilderSettings = mrmeshpy.MeshBuilder.BuildSettings
    mrmeshpy.MeshToVolumeParamsType = mrmeshpy.MeshToVolumeParams.Type
    mrmeshpy.ObjectDistanceMap.extractDistanceMap = mrmeshpy.ObjectDistanceMap.getDistanceMap
    mrmeshpy.ObjectLines.extractLines = mrmeshpy.ObjectLines.polyline
    mrmeshpy.ObjectMesh.extractMesh = mrmeshpy.ObjectMesh.mesh
    mrmeshpy.ObjectPoints.extractPoints = mrmeshpy.ObjectPoints.pointCloud
    mrmeshpy.objectSave = mrmeshpy.ObjectSave.toAnySupportedFormat
    mrmeshpy.ObjectVoxels.extractVoxels = mrmeshpy.ObjectVoxels.vdbVolume
    mrmeshpy.saveAllSlicesToImage = mrmeshpy.VoxelsSave.saveAllSlicesToImage
    mrmeshpy.saveMesh = mrmeshpy.MeshSave.toAnySupportedFormat
    mrmeshpy.savePoints = mrmeshpy.PointsSave.toAnySupportedFormat
    mrmeshpy.saveSliceToImage = mrmeshpy.VoxelsSave.saveSliceToImage
    mrmeshpy.saveVoxelsGav = mrmeshpy.VoxelsSave.toGav
    mrmeshpy.saveVoxelsRaw = mrmeshpy.VoxelsSave.toRawAutoname
    mrmeshpy.TextAlignParams = mrmeshpy.TextMeshAlignParams
    mrmeshpy.topologyFromTriangles = mrmeshpy.MeshBuilder.fromTriangles
    mrmeshpy.triangulateContours = mrmeshpy.PlanarTriangulation.triangulateContours
    mrmeshpy.Triangulation.vec = mrmeshpy.Triangulation.vec_
    mrmeshpy.uniteCloseVertices = mrmeshpy.MeshBuilder.uniteCloseVertices
    mrmeshpy.vectorConstMeshPtr = mrmeshpy.std_vector_const_Mesh
    mrmeshpy.vectorEdges = mrmeshpy.EdgeLoop
    mrmeshpy.VertCoords.vec = mrmeshpy.VertCoords.vec_
    mrmeshpy.VertScalars.vec = mrmeshpy.VertScalars.vec_
    mrmeshpy.VoxelsSaveSavingSettings = mrmeshpy.VoxelsSave.SavingSettings
    mrmeshpy.TriangulationHelpersSettings = mrmeshpy.TriangulationHelpers.Settings
    mrmeshpy.buildUnitedLocalTriangulations = mrmeshpy.TriangulationHelpers.buildUnitedLocalTriangulations

_init_patch()
del _init_patch

# manually appended from scripts/wheel/init.py

def _override_resources_dir():
    """
    override resources directory to the package's dir
    """
    import pathlib
    from . import mrmeshpy as mr

    mr.SystemPath.overrideDirectory(mr.SystemPath.Directory.Resources, pathlib.Path(__file__).parent.resolve())
    mr.SystemPath.overrideDirectory(mr.SystemPath.Directory.Fonts, pathlib.Path(__file__).parent.resolve())

_override_resources_dir()
del _override_resources_dir
