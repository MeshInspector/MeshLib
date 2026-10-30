"""Conversions between MeshLib objects and NumPy arrays.

The data is copied in bulk through NumPy views of the memory of MeshLib containers (`data_pointer()`).
"""

import os
from concurrent.futures import ThreadPoolExecutor

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
]


class _Memory:
    """Exposes the memory of `vec` to NumPy, and keeps `vec` alive while it is used."""
    def __init__(self, vec, typestr, shape):
        self.__array_interface__ = {'data': (vec.data_pointer(), False), 'typestr': typestr, 'shape': shape, 'version': 3}
        self.vec = vec


def _view(vec, typestr, shape):
    """A NumPy array sharing the memory of `std::vector` `vec` (or of `MR::Vector` over one)."""
    if not hasattr(vec, 'data_pointer'):
        vec = vec.vec
    return np.asarray(_Memory(vec, typestr, shape))


_pool = None
_threads = os.cpu_count() or 1


def _parallel(n, f, min_chunk):
    """Calls `f(begin, end)` for chunks of `range(n)` of at least `min_chunk`, in threads if more than one, and returns the results."""
    global _pool
    chunks = max(1, min(_threads, n // max(1, min_chunk)))
    if chunks == 1:
        return [f(0, n)]
    if _pool is None:
        _pool = ThreadPoolExecutor(_threads)
    step = -(-n // chunks)
    return list(_pool.map(lambda b: f(b, min(b + step, n)), range(0, n, step)))


# elements per thread, below which the threads cost more than they save
_MIN_CHUNK = 1 << 18


def _copy(dst, src):
    """`dst[...] = src` for arrays of the same shape, by slices of the first axis in threads: NumPy releases the GIL while copying."""
    def copy(b, e):
        dst[b:e] = src[b:e]
    _parallel(len(dst), copy, _MIN_CHUNK * len(dst) // max(1, dst.size))


def _copied(src):
    res = np.empty(src.shape, dtype=src.dtype)
    _copy(res, src)
    return res


def _check_float(a):
    if a.dtype != np.float64 and a.dtype != np.float32:
        raise RuntimeError("dtype of input python vector should be float32 or float64")


def _check_coords(arr, name):
    a = np.asarray(arr)
    if a.ndim != 2 or a.shape[1] != 3:
        raise RuntimeError("shape of input python vector '" + name + "' should be (n,3)")
    _check_float(a)
    return a


def _fill(vec, typestr, a):
    """Resizes `vec` to the rows of `a` and copies them there."""
    vec.resize(a.shape[0])
    _copy(_view(vec, typestr, a.shape), a)


def _bools(bitset, size):
    """The first `size` bits of `bitset`, `False` past its end."""
    blocks = bitset.bits()
    bits = np.unpackbits(_view(blocks, '<u1', (8 * blocks.size(),)), count=min(size, bitset.size()), bitorder='little').view(bool)
    if size > bits.size:
        bits = np.concatenate([bits, np.zeros(size - bits.size, dtype=bool)])
    return bits


_Blocks = type(_mr.BitSet().bits())


def _bitset_from_bools(cls, bools):
    """A `cls` bit set with the bits of the flat bool array `bools`."""
    packed = np.packbits(bools, bitorder='little')
    blocks = _Blocks()
    blocks.resize((packed.size + 7) // 8)
    _view(blocks, '<u1', (packed.size,))[:] = packed
    res = _mr.BitSet.fromBlocks(blocks)
    res.resize(bools.size)
    return cls(res)


def meshFromFacesVerts(faces, verts, settings=_mr.MeshBuilder.BuildSettings(), duplicateNonManifoldVertices=True):
    """meshFromFacesVerts(faces: Buffer, verts: Buffer, settings: meshlib.mrmeshpy.MeshBuilder.BuildSettings = MeshBuilderSettings(), duplicateNonManifoldVertices: bool = True) -> meshlib.mrmeshpy.Mesh

constructs mesh from given numpy ndarrays of faces (N VertId x3, FaceId x1), verts (M vec3 x3)
"""
    f = np.asarray(faces)
    if f.ndim != 2 or f.shape[1] != 3:
        raise RuntimeError("shape of input python vector 'faces' should be (n,3)")
    v = _check_coords(verts, 'verts')
    if f.itemsize != 4 and f.itemsize != 8:
        raise RuntimeError("dtype of input python vector 'faces' should be int32 or int64")
    t = _mr.Triangulation()
    _fill(t, '<i4', f)
    points = _mr.VertCoords()
    _fill(points, '<f4', v)
    # the points are swapped in after building the mesh: passing them would copy them
    if not duplicateNonManifoldVertices:
        res = _mr.Mesh.fromTriangles(_mr.VertCoords(), t, settings)
    else:
        dups = _mr.std_vector_MeshBuilder_VertDuplication()
        res = _mr.Mesh.fromTrianglesDuplicatingNonManifoldVertices(_mr.VertCoords(), t, dups, settings)
        if points.size() < res.points.size():
            points.resize(res.points.size())
        if dups.size():
            d = _view(dups, '<i4', (dups.size(), 2))
            p = _view(points, '<f4', (points.size(), 3))
            p[d[:, 1]] = p[d[:, 0]]
    res.points.swap(points)
    return res


def meshFromUVPoints(xArray, yArray, zArray):
    """meshFromUVPoints(xArray: Buffer, yArray: Buffer, zArray: Buffer) -> meshlib.mrmeshpy.Mesh

constructs mesh from three 2d numpy ndarrays with x,y,z positions of mesh
"""
    arrays = []
    for a, name in ((xArray, 'X'), (yArray, 'Y'), (zArray, 'Z')):
        a = np.asarray(a)
        if a.ndim != 2:
            raise RuntimeError(name + " should be 2D")
        if arrays and a.shape != arrays[0].shape:
            raise RuntimeError("Input arrays shapes should be same")
        if a.dtype != np.float32 and a.dtype != np.float64:
            raise RuntimeError(name + " dtype should be float32 or float64")
        if arrays and a.dtype != arrays[0].dtype:
            raise RuntimeError("Arrays should have same dtype")
        arrays.append(a)
    sx, sy = arrays[0].shape

    # point v * sx + u is at [u, v] of the arrays
    points = _mr.VertCoords()
    points.resize(sx * sy)
    p = _view(points, '<f4', (sy, sx, 3))
    for c, a in enumerate(arrays):
        _copy(p[:, :, c], a.T)

    old_to_new = _mr.findSmallestCloseVertices(points, float(np.finfo(np.float32).eps))
    w = _view(old_to_new, '<i4', (sy, sx))

    # cell [i, j] (row-major) gives F = (i,j),(i,j+1),(i+1,j) if i < sy-1 and j < sx-1, then S = (i,j),(i,j-1),(i-1,j) if i > 0 and j > 0
    t = _mr.Triangulation()
    n = 2 * (sx - 1) * (sy - 1) if sx and sy else 0
    t.resize(n)
    if n:
        tv = _view(t, '<i4', (n, 3))
        first, last = tv[:sx - 1], tv[n - (sx - 1):]
        first[:, 0], first[:, 1], first[:, 2] = w[0, :-1], w[0, 1:], w[1, :-1]
        last[:, 0], last[:, 1], last[:, 2] = w[-1, 1:], w[-1, :-1], w[-2, 1:]
        # a middle row i: F(i,0), then F(i,j), S(i,j) for 0 < j < sx-1, then S(i,sx-1)
        mid = tv[sx - 1:n - (sx - 1)].reshape(sy - 2, sx - 1, 2, 3)

        def rows(b, e):
            m, wi, wu, wd = mid[b:e], w[b + 1:e + 1], w[b:e], w[b + 2:e + 2]
            m[:, 0, 0, 0], m[:, 0, 0, 1], m[:, 0, 0, 2] = wi[:, 0], wi[:, 1], wd[:, 0]
            m[:, 1:, 0, 0], m[:, 1:, 0, 1], m[:, 1:, 0, 2] = wi[:, 1:-1], wi[:, :-2], wu[:, 1:-1]
            m[:, :-1, 1, 0], m[:, :-1, 1, 1], m[:, :-1, 1, 2] = wi[:, 1:-1], wi[:, 2:], wd[:, 1:-1]
            m[:, -1, 1, 0], m[:, -1, 1, 1], m[:, -1, 1, 2] = wi[:, -1], wi[:, -2], wu[:, -1]
        _parallel(sy - 2, rows, _MIN_CHUNK // (6 * sx))

    res = _mr.Mesh.fromTriangles(_mr.VertCoords(), t)
    res.points.swap(points)
    if res.volume() < 0.0:
        res.topology.flipOrientation()
    res.pack()
    return res


def pointCloudFromPoints(points, normals=np.array([])):
    """pointCloudFromPoints(points: Buffer, normals: Buffer = array([], dtype=float64)) -> meshlib.mrmeshpy.PointCloud

creates point cloud object from numpy arrays, first arg - points, second optional arg - normals
"""
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] != 3:
        raise RuntimeError("shape of input python vector 'points' should be (n,3)")
    n = np.asarray(normals)
    if n.size != 0 and (n.ndim != 2 or n.shape[1] != 3):
        raise RuntimeError("shape of input python vector 'normals' should be (n,3) or empty")
    _check_float(p)
    res = _mr.PointCloud()
    _fill(res.points, '<f4', p)
    if n.size > 0:
        _check_float(n)
        _fill(res.normals, '<f4', n)
    res.validPoints.resize(p.shape[0], True)
    return res


def polyline2FromPoints(points):
    """polyline2FromPoints(points: Buffer) -> meshlib.mrmeshpy.Polyline2

creates polyline2 object from numpy array
"""
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] != 2:
        raise RuntimeError("shape of input python vector 'points' should be (n,2)")
    _check_float(p)
    contour = _mr.Contour2f()
    _fill(contour, '<f4', p)
    return _mr.Polyline2(contour)


def getNumpyFaces(topology):
    """getNumpyFaces(topology: meshlib.mrmeshpy.MeshTopology) -> numpy.ndarray[numpy.int32]

returns numpy array shapes [num faces,3] which represents vertices of mesh valid faces
"""
    num_faces = int(topology.lastValidFace()) + 1
    t = topology.getTriangulation()
    res = _view(t, '<i4', (t.size(), 3))[:num_faces]
    if topology.numValidFaces() != num_faces:
        res[~_bools(topology.getValidFaces(), num_faces)] = 0
    return res


def toNumpyArray(coords):
    """toNumpyArray(*args, **kwargs)
Overloaded function.

1. toNumpyArray(coords: meshlib.mrmeshpy.VertCoords) -> numpy.ndarray[numpy.float32]

returns numpy array shapes [num coords,3] which represents coordinates from given vector

2. toNumpyArray(coords: meshlib.mrmeshpy.FaceNormals) -> numpy.ndarray[numpy.float32]

returns numpy array shapes [num coords,3] which represents coordinates from given vector

3. toNumpyArray(coords: meshlib.mrmeshpy.std_vector_Vector3_float) -> numpy.ndarray[numpy.float32]

returns numpy array shapes [num coords,3] which represents coordinates from given vector
"""
    if not isinstance(coords, (_mr.VertCoords, _mr.FaceNormals, _mr.std_vector_Vector3_float)):
        raise TypeError("toNumpyArray(): incompatible function arguments")
    return _copied(_view(coords, '<f4', (coords.size(), 3)))


def getNumpyVerts(mesh):
    """getNumpyVerts(mesh: meshlib.mrmeshpy.Mesh) -> numpy.ndarray[numpy.float32]

returns numpy array shapes [num verts,3] which represents coordinates of all mesh points (including invalid ones)
"""
    return toNumpyArray(mesh.points)


def getNumpyBitSet(bitset):
    """getNumpyBitSet(bitset: meshlib.mrmeshpy.BitSet) -> numpy.ndarray[bool]

returns numpy array with bools for each bit of given bitset
"""
    return _bools(bitset, bitset.size())


def _per_vertex(mesh, scalars):
    """`scalars` (0 in invalid vertices) up to the last valid vertex, sharing their memory."""
    return _view(scalars, '<f4', (scalars.size(),))[:int(mesh.topology.lastValidVert()) + 1]


def getNumpyMeanCurvature(mesh):
    """getNumpyMeanCurvature(mesh: meshlib.mrmeshpy.Mesh) -> numpy.ndarray[numpy.float32]

retunrs numpy array with discrete mean curvature for each vertex of a mesh
"""
    return _per_vertex(mesh, mesh.discreteMeanCurvatures())


def getNumpyCurvature(mesh):
    """getNumpyCurvature(mesh: meshlib.mrmeshpy.Mesh) -> numpy.ndarray[numpy.float32]

retunrs numpy array with discrete mean curvature for each vertex of a mesh
"""
    return getNumpyMeanCurvature(mesh)


def getNumpyGaussianCurvature(mesh):
    """getNumpyGaussianCurvature(mesh: meshlib.mrmeshpy.Mesh) -> numpy.ndarray[numpy.float32]

retunrs numpy array with discrete Gaussian curvature for each vertex of a mesh
"""
    return _per_vertex(mesh, mesh.discreteGaussianCurvatures())


def fromNumpyArray(coords):
    """fromNumpyArray(coords: Buffer) -> meshlib.mrmeshpy.VertCoords

constructs mrmeshpy.VertCoords from numpy ndarray with shape (n,3)
"""
    res = _mr.VertCoords()
    _fill(res, '<f4', _check_coords(coords, 'coords'))
    return res


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
    """faceBitSetFromBools(boolArray: Buffer) -> meshlib.mrmeshpy.FaceBitSet

returns FaceBitSet from numpy array with bools
"""
    return _bitset_from_np(_mr.FaceBitSet, boolArray)


def vertBitSetFromBools(boolArray):
    """vertBitSetFromBools(boolArray: Buffer) -> meshlib.mrmeshpy.VertBitSet

returns VertBitSet from numpy array with bools
"""
    return _bitset_from_np(_mr.VertBitSet, boolArray)


def edgeBitSetFromBools(boolArray):
    """edgeBitSetFromBools(boolArray: Buffer) -> meshlib.mrmeshpy.EdgeBitSet

returns EdgeBitSet from numpy array with bools
"""
    return _bitset_from_np(_mr.EdgeBitSet, boolArray)


def undirectedEdgeBitSetFromBools(boolArray):
    """undirectedEdgeBitSetFromBools(boolArray: Buffer) -> meshlib.mrmeshpy.UndirectedEdgeBitSet

returns UndirectedEdgeBitSet from numpy array with bools
"""
    return _bitset_from_np(_mr.UndirectedEdgeBitSet, boolArray)


def voxelBitSetFromBools(boolArray):
    """voxelBitSetFromBools(boolArray: Buffer) -> meshlib.mrmeshpy.VoxelBitSet

returns VoxelBitSet from numpy array with bools
"""
    return _bitset_from_np(_mr.VoxelBitSet, boolArray)


# Voxels: `VolumeIndexer` has `x` fastest, while the NumPy arrays have the shape (x,y,z).
if hasattr(_mr, 'SimpleVolumeMinMax'):
    __all__ += ['simpleVolumeFrom3Darray', 'voxelBitSetFrom3Darray', 'getNumpy3Darray']

    def simpleVolumeFrom3Darray(*args, **kwargs):
        """simpleVolumeFrom3Darray(3DvoxelsArray: Buffer) -> meshlib.mrmeshpy.SimpleVolumeMinMax

Convert numpy 3D array to SimpleVolume
"""
        a = np.asarray(args[0] if args else kwargs['3DvoxelsArray'])
        if a.ndim != 3:
            raise RuntimeError("shape of input python vector 'voxelsArray' should be (x,y,z)")
        _check_float(a)
        res = _mr.SimpleVolumeMinMax()
        res.dims = _mr.Vector3i(*a.shape)
        res.data.resize(a.size)
        if a.size:
            dst, src = _view(res.data, '<f4', a.shape[::-1]), a.T

            def copy(b, e):
                dst[b:e] = src[b:e]
                return dst[b:e].min(), dst[b:e].max()
            minmax = _parallel(len(dst), copy, _MIN_CHUNK * len(dst) // dst.size)
            res.min = float(min(m[0] for m in minmax))
            res.max = float(max(m[1] for m in minmax))
        return res

    def voxelBitSetFrom3Darray(boolsArray):
        """voxelBitSetFrom3Darray(boolsArray: Buffer) -> meshlib.mrmeshpy.VoxelBitSet

Convert numpy 3D array of bools with shape (x,y,z) to VoxelBitSet indexed as in VolumeIndexer
"""
        a = np.asarray(boolsArray)
        if a.ndim != 3:
            raise RuntimeError("shape of input python vector 'boolsArray' should be (x,y,z)")
        if a.dtype != np.bool_:
            raise RuntimeError("dtype of input python vector should be bool")
        bools = np.empty(a.shape[::-1], dtype=bool)
        _copy(bools, a.T)
        return _bitset_from_bools(_mr.VoxelBitSet, bools.ravel())

    def getNumpy3Darray(*args, **kwargs):
        """getNumpy3Darray(*args, **kwargs)
Overloaded function.

1. getNumpy3Darray(simpleVolume: meshlib.mrmeshpy.SimpleVolume) -> numpy.ndarray[numpy.float32]

Convert SimpleVolume to numpy 3D array

2. getNumpy3Darray(voxelBitSet: meshlib.mrmeshpy.VoxelBitSet, dims: meshlib.mrmeshpy.Vector3i) -> numpy.ndarray[bool]

Convert VoxelBitSet of the volume with given dimensions to numpy 3D array of bools with shape (x,y,z)
"""
        if len(args) + len(kwargs) == 1:
            volume = args[0] if args else kwargs['simpleVolume']
            d = volume.dims
            return _copied(_view(volume.data, '<f4', (d.z, d.y, d.x)).T)
        bitset = args[0] if args else kwargs['voxelBitSet']
        d = args[1] if len(args) > 1 else kwargs['dims']
        size = d.x * d.y * d.z
        if bitset.size() > size:
            raise RuntimeError("bitSet is larger than the volume with given dimensions")
        res = np.empty((d.x, d.y, d.z), dtype=bool)
        _copy(res, _bools(bitset, size).reshape(d.z, d.y, d.x).T)
        return res
