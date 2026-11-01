# Python helper functions of `mrmeshpy`, which is generated over the C bindings (see `generate_python.sh`).
# Runs in the namespace of `mrmeshpy` when it's imported (`mrbind_gen_python --extra-python`).

def fixSelfIntersections(mesh, voxelSize):
    convert = MeshVoxelsConverter()
    convert.voxelSize = voxelSize
    gridA = convert(mesh)
    mesh.assign(convert(gridA))

def voxelBooleanSubtract(mesh1, mesh2, voxelSize):
    convert = MeshVoxelsConverter()
    convert.voxelSize = voxelSize
    gridA = convert(mesh1)
    gridA -= convert(mesh2)
    return convert(gridA)

def voxelBooleanUnite(mesh1, mesh2, voxelSize):
    convert = MeshVoxelsConverter()
    convert.voxelSize = voxelSize
    gridA = convert(mesh1)
    gridA += convert(mesh2)
    return convert(gridA)

def voxelBooleanIntersect(mesh1, mesh2, voxelSize):
    convert = MeshVoxelsConverter()
    convert.voxelSize = voxelSize
    gridA = convert(mesh1)
    gridA *= convert(mesh2)
    return convert(gridA)

def projectAllMeshVertices(refMesh, mesh, refXf=None, xf=None, upDistLimitSq=3.4028234663852886e+38, loDistLimitSq=0.0):
    params = MeshProjectionParameters()
    params.loDistLimitSq = loDistLimitSq
    params.upDistLimitSq = upDistLimitSq
    params.refXf = refXf
    params.xf = xf
    res = findSignedDistances(refMesh, mesh, params, PointsToMeshProjector())
    res.resize(mesh.topology.lastValidVert().get() + 1)
    return res

def mergeMeshes(meshes):
    res = Mesh()
    for m in meshes:
        res.addMesh(m)
    return res

def getFacesByMinEdgeLength(mesh, minLength):
    topology = mesh.topology
    resultFaces = FaceBitSet(topology.getValidFaces().size())
    minLengthSq = minLength * minLength
    for ue in undirectedEdges(topology):
        if mesh.edgeLengthSq(ue) > minLengthSq:
            l = topology.left(ue)
            r = topology.right(ue)
            if l.valid():
                resultFaces.set(l)
            if r.valid():
                resultFaces.set(r)
    return resultFaces

def loadSceneObject(path, callback=None):
    settings = SceneLoad.Settings()
    if callback is not None:
        settings.progress = callback
    result = SceneLoad.fromAnySupportedFormat([path], settings)
    if not result.scene or result.errorSummary:
        raise RuntimeError(result.errorSummary)
    if not result.isSceneConstructed or len(result.scene.children()) != 1:
        return result.scene
    return result.scene.children()[0]

def saveDistanceMapToImage(distMap, filename, threshold=0.003921568859368563):
    ImageSave.toAnySupportedFormat(convertDistanceMapToImage(distMap, threshold), filename)

def loadDistanceMapFromImage(filename, threshold=0.003921568859368563):
    return convertImageToDistanceMap(ImageLoad.fromAnySupportedFormat(filename), threshold)

def _setDocs():
    # Like the pybind11 docs of the C++ functions.
    m = "meshlib.mrmeshpy."
    path = "Union[os.PathLike, str, bytes]"
    docs = {
        fixSelfIntersections: f"fixSelfIntersections(mesh: {m}Mesh, voxelSize: float) -> None\n\nFix self-intersections by converting to voxels and back.\n",
        voxelBooleanSubtract: f"voxelBooleanSubtract(mesh1: {m}Mesh, mesh2: {m}Mesh, voxelSize: float) -> {m}Mesh\n\nSubtract mesh B from mesh A.\n",
        voxelBooleanUnite: f"voxelBooleanUnite(mesh1: {m}Mesh, mesh2: {m}Mesh, voxelSize: float) -> {m}Mesh\n\nUnite mesh A and mesh B.\n",
        voxelBooleanIntersect: f"voxelBooleanIntersect(mesh1: {m}Mesh, mesh2: {m}Mesh, voxelSize: float) -> {m}Mesh\n\nIntersect mesh A and mesh B.\n",
        projectAllMeshVertices: f"projectAllMeshVertices(refMesh: {m}Mesh, mesh: {m}Mesh, refXf: {m}AffineXf3f = None, xf: {m}AffineXf3f = None, upDistLimitSq: float = 3.4028234663852886e+38, loDistLimitSq: float = 0.0) -> {m}VertScalars\n\n"
            "Computes signed distances from all mesh points to refMesh.\n"
            "`refMesh` - all points will me projected to this mesh\n"
            "`mesh` - this mesh points will be projected\n"
            "`refXf` - world transform for refMesh\n"
            "`upDistLimitSq` - upper limit on the distance in question, if the real distance is larger than the returning upDistLimit\n"
            "`loDistLimitSq` - low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one\n",
        mergeMeshes: f"mergeMeshes(meshes: {m}std_vector_std_shared_ptr_Mesh) -> {m}Mesh\n\nMerge a list of meshes to one mesh.\n",
        getFacesByMinEdgeLength: f"getFacesByMinEdgeLength(mesh: {m}Mesh, minLength: float) -> {m}FaceBitSet\n\nReturn faces with at least one edge longer than the specified length.\n",
        loadSceneObject: f"loadSceneObject(path: {path}, callback: {m}func_bool_from_float = '{{}}') -> {m}Object\n\nDetects the format from file extension and loads scene object from it.\n",
        saveDistanceMapToImage: f"saveDistanceMapToImage(distMap: {m}DistanceMap, filename: {path}, threshold: float = 0.003921568859368563) -> None\n\n"
            "saves distance map to a grayscale image file\n"
            "    threshold - threshold of maximum values [0.; 1.]. invalid pixel set as 0. (black)\n"
            "minimum (close): 1.0 (white)\n"
            "maximum (far): threshold\n"
            "invalid (infinity): 0.0 (black)\n",
        loadDistanceMapFromImage: f"loadDistanceMapFromImage(filename: {path}, threshold: float = 0.003921568859368563) -> {m}DistanceMap\n\n"
            "load distance map from a grayscale image file\n"
            "    threshold - threshold of valid values [0.; 1.]. pixel with color less then threshold set invalid\n",
    }
    for func, doc in docs.items():
        func.__doc__ = doc

_setDocs()
del _setDocs
