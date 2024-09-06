# Fixes DLL loading on Windows.

def _init_patch():
    import os
    libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    os.add_dll_directory(libs_dir)

_init_patch()
del _init_patch



# Manually define some aliases. It's recommended to avoid those.

def _init_patch():
    from meshlib2 import mrmeshpy as mrmesh
    mrmesh.getAllComponentsVerts = mrmesh.MeshComponents_getAllComponentsVerts
    mrmesh.BooleanResMapObj = mrmesh.BooleanResultMapper_MapObject
    mrmesh.vectorConstMeshPtr = mrmesh.std_vector_const_Mesh
    mrmesh.vectorEdges = mrmesh.EdgeLoop
    mrmesh.copyMesh = mrmesh.Mesh
    mrmesh.localFindSelfIntersections = mrmesh.SelfIntersections_getFaces
    mrmesh.localFixSelfIntersections = mrmesh.SelfIntersections_fix
    mrmesh.FixSelfIntersectionSettings = mrmesh.SelfIntersections_Settings
    mrmesh.FixSelfIntersectionMethod = mrmesh.SelfIntersections_Settings_Method
    mrmesh.triangulateContours = mrmesh.PlanarTriangulation_triangulateContours
    mrmesh.TextAlignParams = mrmesh.TextMeshAlignParams
    mrmesh.MeshToVolumeParamsType = mrmesh.MeshToVolumeParams_Type
    mrmesh.VoxelsSaveSavingSettings = mrmesh.VoxelsSave_SavingSettings
    mrmesh.saveMesh = mrmesh.MeshSave_toAnySupportedFormat # There was also a manually written overload, hmm.
    mrmesh.loadMesh = mrmesh.MeshLoad_fromAnySupportedFormat
    mrmesh.saveSliceToImage = mrmesh.VoxelsSave_saveSliceToImage
    mrmesh.saveAllSlicesToImage = mrmesh.VoxelsSave_saveAllSlicesToImage
    mrmesh.LaplacianEdgeWeightsParam = mrmesh.EdgeWeights
    mrmesh.getAllComponents = mrmesh.MeshComponents_getAllComponents
    mrmesh.MeshBuilderSettings = mrmesh.MeshBuilder_BuildSettings
    mrmesh.findUndercuts = mrmesh.FixUndercuts_findUndercuts
    mrmesh.fixUndercuts = mrmesh.FixUndercuts_fixUndercuts
    mrmesh.topologyFromTriangles = mrmesh.MeshBuilder_fromTriangles
    mrmesh.uniteCloseVertices = mrmesh.MeshBuilder_uniteCloseVertices

_init_patch()
del _init_patch
