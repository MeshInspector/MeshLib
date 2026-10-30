"""`mrmeshnumpy` for the `mrmeshpy` generated over the C bindings (see `python_over_c.sh`), installed as `meshlib/_mrmeshnumpy.py`.

The same functions as the pybind11 module in `source/mrmeshnumpy`, in Python: the data is copied
between NumPy and the MeshLib containers in bulk, through their `data_pointer()`.
`mrmeshnumpy.py` star-imports this, so that only the public functions show up there.
"""

import ctypes

import numpy as np

from . import mrmeshpy as _mr

__all__ = [
    'meshFromFacesVerts',
    'meshFromUVPoints',
    'pointCloudFromPoints',
    'polyline2FromPoints',
    'getNumpyFaces',
    'toNumpyArray',
    'getNumpyVerts',
    'getNumpyBitSet',
    'getNumpyMeanCurvature',
    'getNumpyCurvature',
    'getNumpyGaussianCurvature',
    'fromNumpyArray',
    'faceBitSetFromBools',
    'vertBitSetFromBools',
    'edgeBitSetFromBools',
    'undirectedEdgeBitSetFromBools',
    'voxelBitSetFromBools',
    'simpleVolumeFrom3Darray',
    'voxelBitSetFrom3Darray',
    'getNumpy3Darray',
]


def _view(vec, ctype, count):
    """A NumPy view of the contiguous `std::vector` (or an `MR::Vector` over one) `vec`, as `count` values of `ctype`."""
    if not hasattr(vec, 'data_pointer'):
        vec = vec.vec
    if count == 0:
        return np.ctypeslib.as_array((ctype * 1)())[:0]
    return np.ctypeslib.as_array((ctype * count).from_address(vec.data_pointer()))


def _coords(arr, name):
    a = np.asarray(arr)
    if a.ndim != 2 or a.shape[1] != 3:
        raise RuntimeError("shape of input python vector '" + name + "' should be (n,3)")
    return a


def _to_vert_coords(a):
    if a.dtype != np.float64 and a.dtype != np.float32:
        raise RuntimeError("dtype of input python vector should be float32 or float64")
    res = _mr.VertCoords()
    res.resize(a.shape[0])
    _view(res.vec, ctypes.c_float, 3 * a.shape[0]).reshape(-1, 3)[:] = a
    return res


def _bools(bitset, size=None):
    """The bits of `bitset` as bools, padded with `False` to `size`."""
    blocks = bitset.bits()
    bits = np.unpackbits(_view(blocks, ctypes.c_uint64, blocks.size()).view(np.uint8), bitorder='little')[:bitset.size()].astype(bool)
    if size is not None and size > bits.size:
        bits = np.concatenate([bits, np.zeros(size - bits.size, dtype=bool)])
    return bits


def _bitset_from_bools(cls, bools):
    packed = np.packbits(bools, bitorder='little')
    packed = np.concatenate([packed, np.zeros(-packed.size % 8, dtype=np.uint8)]).view(np.uint64)
    blocks = type(_mr.BitSet().bits())()
    blocks.resize(packed.size)
    _view(blocks, ctypes.c_uint64, packed.size)[:] = packed
    res = _mr.BitSet.fromBlocks(blocks)
    res.resize(bools.size)
    return cls(res)


def meshFromFacesVerts(faces, verts, settings=_mr.MeshBuilder.BuildSettings(), duplicateNonManifoldVertices=True):
    f = np.asarray(faces)
    if f.ndim != 2 or f.shape[1] != 3:
        raise RuntimeError("shape of input python vector 'faces' should be (n,3)")
    v = _coords(verts, 'verts')
    if f.itemsize != 4 and f.itemsize != 8:
        raise RuntimeError("dtype of input python vector 'faces' should be int32 or int64")
    t = _mr.Triangulation()
    t.resize(f.shape[0])
    _view(t.vec, ctypes.c_int32, 3 * f.shape[0]).reshape(-1, 3)[:] = f
    points = _to_vert_coords(v)
    if duplicateNonManifoldVertices:
        return _mr.Mesh.fromTrianglesDuplicatingNonManifoldVertices(points, t, None, settings)
    return _mr.Mesh.fromTriangles(points, t, settings)


def meshFromUVPoints(xArray, yArray, zArray):
    arrays = []
    shape = None
    dtype = None
    for a, name in ((xArray, 'X'), (yArray, 'Y'), (zArray, 'Z')):
        a = np.asarray(a)
        if a.ndim != 2:
            raise RuntimeError(name + " should be 2D")
        if shape is None:
            shape = a.shape
        elif shape != a.shape:
            raise RuntimeError("Input arrays shapes should be same")
        if a.dtype != np.float32 and a.dtype != np.float64:
            raise RuntimeError(name + " dtype should be float32 or float64")
        if dtype is None:
            dtype = a.dtype
        elif dtype != a.dtype:
            raise RuntimeError("Arrays should have same dtype")
        arrays.append(np.ascontiguousarray(a))
    sx, sy = shape

    # Point `i` is at `[i % sx, i // sx]`.
    res = _mr.Mesh()
    res.points = _to_vert_coords(np.stack([a.T.ravel() for a in arrays], axis=1))

    # Two triangles per cell, in the order of the cells.
    i, j = np.meshgrid(np.arange(sy), np.arange(sx), indexing='ij')
    i, j = i.ravel(), j.ravel()
    v = i * sx + j
    first = np.stack([v, v + 1, v + sx], axis=1)
    second = np.stack([v, v - 1, v - sx], axis=1)
    mask = np.stack([(i + 1 < sy) & (j + 1 < sx), (i > 0) & (j > 0)], axis=1)
    tris = np.stack([first, second], axis=1)[mask]

    # Unite the close vertices.
    old_to_new = _mr.findSmallestCloseVertices(res.points, float(np.finfo(np.float32).eps))
    tris = _view(old_to_new.vec, ctypes.c_int32, old_to_new.size())[tris]
    t = _mr.Triangulation()
    t.resize(tris.shape[0])
    _view(t.vec, ctypes.c_int32, 3 * tris.shape[0]).reshape(-1, 3)[:] = tris
    res.topology = _mr.MeshBuilder.fromTriangles(t)

    if res.volume() < 0.0:
        res.topology.flipOrientation()
    res.pack()
    return res


def pointCloudFromPoints(points, normals=np.array([])):
    p = _coords(points, 'points')
    n = np.asarray(normals)
    if n.size != 0 and (n.ndim != 2 or n.shape[1] != 3):
        raise RuntimeError("shape of input python vector 'normals' should be (n,3) or empty")
    res = _mr.PointCloud()
    res.points = _to_vert_coords(p)
    if n.size > 0:
        res.normals = _to_vert_coords(n)
    res.validPoints = _mr.VertBitSet(p.shape[0], True)
    return res


def polyline2FromPoints(points):
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] != 2:
        raise RuntimeError("shape of input python vector 'points' should be (n,2)")
    if p.dtype != np.float64 and p.dtype != np.float32:
        raise RuntimeError("dtype of input python vector should be float32 or float64")
    contour = _mr.Contour2f()
    contour.resize(p.shape[0])
    _view(contour, ctypes.c_float, 2 * p.shape[0]).reshape(-1, 2)[:] = p
    contours = _mr.Contours2f()
    contours.append(contour)
    return _mr.Polyline2(contours)


def getNumpyFaces(topology):
    num_faces = int(topology.lastValidFace()) + 1
    t = topology.getTriangulation()
    res = _view(t.vec, ctypes.c_int32, 3 * t.size()).reshape(-1, 3)[:num_faces].astype(np.int32)
    res[~_bools(topology.getValidFaces(), num_faces)[:num_faces]] = 0
    return res


def toNumpyArray(coords):
    vec = coords if isinstance(coords, _mr.std_vector_Vector3_float) else coords.vec
    return _view(vec, ctypes.c_float, 3 * vec.size()).reshape(-1, 3).astype(np.float64)


def getNumpyVerts(mesh):
    return toNumpyArray(mesh.points)


def getNumpyBitSet(bitset):
    return _bools(bitset)


def _per_vertex(mesh, scalars):
    """`scalars` (`VertScalars`, 0 in the invalid vertices) up to the last valid vertex, as float64."""
    num_verts = int(mesh.topology.lastValidVert()) + 1
    return _view(scalars, ctypes.c_float, scalars.size())[:num_verts].astype(np.float64)


def getNumpyMeanCurvature(mesh):
    return _per_vertex(mesh, mesh.discreteMeanCurvatures())


def getNumpyCurvature(mesh):
    return getNumpyMeanCurvature(mesh)


def getNumpyGaussianCurvature(mesh):
    return _per_vertex(mesh, mesh.discreteGaussianCurvatures())


def fromNumpyArray(coords):
    return _to_vert_coords(_coords(coords, 'coords'))


def _bitset_from_np(cls, boolArray):
    b = np.asarray(boolArray)
    if b.ndim != 1:
        raise RuntimeError("shape of input python vector 'bools' should be (n)")
    if b.shape[0] == 0:
        return cls()
    if b.dtype != np.bool_:
        raise RuntimeError("format of python vector 'bools' should be bool")
    return _bitset_from_bools(cls, b)


def faceBitSetFromBools(boolArray):
    return _bitset_from_np(_mr.FaceBitSet, boolArray)


def vertBitSetFromBools(boolArray):
    return _bitset_from_np(_mr.VertBitSet, boolArray)


def edgeBitSetFromBools(boolArray):
    return _bitset_from_np(_mr.EdgeBitSet, boolArray)


def undirectedEdgeBitSetFromBools(boolArray):
    return _bitset_from_np(_mr.UndirectedEdgeBitSet, boolArray)


def voxelBitSetFromBools(boolArray):
    return _bitset_from_np(_mr.VoxelBitSet, boolArray)


# Voxels: the `VolumeIndexer` order is `x` fastest, and the NumPy arrays have the shape `(x,y,z)`.

def simpleVolumeFrom3Darray(*args, **kwargs):
    a = np.asarray(args[0] if args else kwargs['3DvoxelsArray'])
    if a.ndim != 3:
        raise RuntimeError("shape of input python vector 'voxelsArray' should be (x,y,z)")
    if a.dtype != np.float64 and a.dtype != np.float32:
        raise RuntimeError("dtype of input python vector should be float32 or float64")
    flat = a.transpose(2, 1, 0).ravel()
    res = _mr.SimpleVolumeMinMax()
    res.dims = _mr.Vector3i(*a.shape)
    res.data.resize(flat.size)
    _view(res.data, ctypes.c_float, flat.size)[:] = flat
    if flat.size:
        res.min = float(flat.astype(np.float32).min())
        res.max = float(flat.astype(np.float32).max())
    return res


def voxelBitSetFrom3Darray(boolsArray):
    a = np.asarray(boolsArray)
    if a.ndim != 3:
        raise RuntimeError("shape of input python vector 'boolsArray' should be (x,y,z)")
    if a.dtype != np.bool_:
        raise RuntimeError("dtype of input python vector should be bool")
    return _bitset_from_bools(_mr.VoxelBitSet, a.transpose(2, 1, 0).ravel())


def getNumpy3Darray(*args, **kwargs):
    if len(args) + len(kwargs) == 1:
        volume = args[0] if args else kwargs['simpleVolume']
        dims = volume.dims
        flat = _view(volume.data, ctypes.c_float, dims.x * dims.y * dims.z)
        return np.ascontiguousarray(flat.reshape(dims.z, dims.y, dims.x).transpose(2, 1, 0), dtype=np.float64)
    bitset = args[0] if args else kwargs['voxelBitSet']
    dims = args[1] if len(args) > 1 else kwargs['dims']
    size = dims.x * dims.y * dims.z
    if bitset.size() > size:
        raise RuntimeError("bitSet is larger than the volume with given dimensions")
    return np.ascontiguousarray(_bools(bitset, size).reshape(dims.z, dims.y, dims.x).transpose(2, 1, 0))


# The docstrings of the pybind11 module, for the same `help()` and API dumps.

edgeBitSetFromBools.__doc__ = 'edgeBitSetFromBools(boolArray: Buffer) -> meshlib.mrmeshpy.EdgeBitSet\n\nreturns EdgeBitSet from numpy array with bools\n'
faceBitSetFromBools.__doc__ = 'faceBitSetFromBools(boolArray: Buffer) -> meshlib.mrmeshpy.FaceBitSet\n\nreturns FaceBitSet from numpy array with bools\n'
fromNumpyArray.__doc__ = 'fromNumpyArray(coords: Buffer) -> meshlib.mrmeshpy.VertCoords\n\nconstructs mrmeshpy.VertCoords from numpy ndarray with shape (n,3)\n'
getNumpy3Darray.__doc__ = 'getNumpy3Darray(*args, **kwargs)\nOverloaded function.\n\n1. getNumpy3Darray(simpleVolume: meshlib.mrmeshpy.SimpleVolume) -> numpy.ndarray[numpy.float64]\n\nConvert SimpleVolume to numpy 3D array\n\n2. getNumpy3Darray(voxelBitSet: meshlib.mrmeshpy.VoxelBitSet, dims: meshlib.mrmeshpy.Vector3i) -> numpy.ndarray[bool]\n\nConvert VoxelBitSet of the volume with given dimensions to numpy 3D array of bools with shape (x,y,z)\n'
getNumpyBitSet.__doc__ = 'getNumpyBitSet(bitset: meshlib.mrmeshpy.BitSet) -> numpy.ndarray[bool]\n\nreturns numpy array with bools for each bit of given bitset\n'
getNumpyCurvature.__doc__ = 'getNumpyCurvature(mesh: meshlib.mrmeshpy.Mesh) -> numpy.ndarray[numpy.float64]\n\nretunrs numpy array with discrete mean curvature for each vertex of a mesh\n'
getNumpyFaces.__doc__ = 'getNumpyFaces(topology: meshlib.mrmeshpy.MeshTopology) -> numpy.ndarray[numpy.int32]\n\nreturns numpy array shapes [num faces,3] which represents vertices of mesh valid faces \n'
getNumpyGaussianCurvature.__doc__ = 'getNumpyGaussianCurvature(mesh: meshlib.mrmeshpy.Mesh) -> numpy.ndarray[numpy.float64]\n\nretunrs numpy array with discrete Gaussian curvature for each vertex of a mesh\n'
getNumpyMeanCurvature.__doc__ = 'getNumpyMeanCurvature(mesh: meshlib.mrmeshpy.Mesh) -> numpy.ndarray[numpy.float64]\n\nretunrs numpy array with discrete mean curvature for each vertex of a mesh\n'
getNumpyVerts.__doc__ = 'getNumpyVerts(mesh: meshlib.mrmeshpy.Mesh) -> numpy.ndarray[numpy.float64]\n\nreturns numpy array shapes [num verts,3] which represents coordinates of all mesh points (including invalid ones)\n'
meshFromFacesVerts.__doc__ = 'meshFromFacesVerts(faces: Buffer, verts: Buffer, settings: meshlib.mrmeshpy.MeshBuilder.BuildSettings = MeshBuilderSettings(), duplicateNonManifoldVertices: bool = True) -> meshlib.mrmeshpy.Mesh\n\nconstructs mesh from given numpy ndarrays of faces (N VertId x3, FaceId x1), verts (M vec3 x3)\n'
meshFromUVPoints.__doc__ = 'meshFromUVPoints(xArray: Buffer, yArray: Buffer, zArray: Buffer) -> meshlib.mrmeshpy.Mesh\n\nconstructs mesh from three 2d numpy ndarrays with x,y,z positions of mesh\n'
pointCloudFromPoints.__doc__ = 'pointCloudFromPoints(points: Buffer, normals: Buffer = array([], dtype=float64)) -> meshlib.mrmeshpy.PointCloud\n\ncreates point cloud object from numpy arrays, first arg - points, second optional arg - normals\n'
polyline2FromPoints.__doc__ = 'polyline2FromPoints(points: Buffer) -> meshlib.mrmeshpy.Polyline2\n\ncreates polyline2 object from numpy array\n'
simpleVolumeFrom3Darray.__doc__ = 'simpleVolumeFrom3Darray(3DvoxelsArray: Buffer) -> meshlib.mrmeshpy.SimpleVolumeMinMax\n\nConvert numpy 3D array to SimpleVolume\n'
toNumpyArray.__doc__ = 'toNumpyArray(*args, **kwargs)\nOverloaded function.\n\n1. toNumpyArray(coords: meshlib.mrmeshpy.VertCoords) -> numpy.ndarray[numpy.float64]\n\nreturns numpy array shapes [num coords,3] which represents coordinates from given vector\n\n2. toNumpyArray(coords: meshlib.mrmeshpy.FaceNormals) -> numpy.ndarray[numpy.float64]\n\nreturns numpy array shapes [num coords,3] which represents coordinates from given vector\n\n3. toNumpyArray(coords: meshlib.mrmeshpy.std_vector_Vector3_float) -> numpy.ndarray[numpy.float64]\n\nreturns numpy array shapes [num coords,3] which represents coordinates from given vector\n'
undirectedEdgeBitSetFromBools.__doc__ = 'undirectedEdgeBitSetFromBools(boolArray: Buffer) -> meshlib.mrmeshpy.UndirectedEdgeBitSet\n\nreturns UndirectedEdgeBitSet from numpy array with bools\n'
vertBitSetFromBools.__doc__ = 'vertBitSetFromBools(boolArray: Buffer) -> meshlib.mrmeshpy.VertBitSet\n\nreturns VertBitSet from numpy array with bools\n'
voxelBitSetFrom3Darray.__doc__ = 'voxelBitSetFrom3Darray(boolsArray: Buffer) -> meshlib.mrmeshpy.VoxelBitSet\n\nConvert numpy 3D array of bools with shape (x,y,z) to VoxelBitSet indexed as in VolumeIndexer\n'
voxelBitSetFromBools.__doc__ = 'voxelBitSetFromBools(boolArray: Buffer) -> meshlib.mrmeshpy.VoxelBitSet\n\nreturns VoxelBitSet from numpy array with bools\n'

for _name in __all__:
    globals()[_name].__module__ = 'meshlib.mrmeshnumpy'
