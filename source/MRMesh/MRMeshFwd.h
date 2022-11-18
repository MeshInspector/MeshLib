#pragma once

// Not zero _ITERATOR_DEBUG_LEVEL in Microsoft STL greatly reduce the performance of STL containers.
// So we change its value to zero by default. A huge restriction with this is that 
// all other linked DLL's and LIBS' also need to define this symbol to remove STL debugging, see
// 1) vcpkg/triplets/x64-windows-meshrus.cmake and
// 2) MeshLib/source/common.props
// If you would like not-zero _ITERATOR_DEBUG_LEVEL and
// you know what you are doing (up to 100x slowdown),
// please define MR_ITERATOR_DEBUG_LEVEL as well
#if defined _MSC_VER
    #if !defined _ITERATOR_DEBUG_LEVEL
        #define _ITERATOR_DEBUG_LEVEL 0
    #endif
    #if !defined MR_ITERATOR_DEBUG_LEVEL
        #define MR_ITERATOR_DEBUG_LEVEL 0
    #endif
    #if _ITERATOR_DEBUG_LEVEL != MR_ITERATOR_DEBUG_LEVEL
        #error _ITERATOR_DEBUG_LEVEL is inconsistent with MeshLib
    #endif
#endif

#include <array>
#include <vector>
#include <parallel_hashmap/phmap_fwd_decl.h>

#ifdef _WIN32
#   ifdef MRMESH_EXPORT
#       define MRMESH_API __declspec(dllexport)
#   else
#       define MRMESH_API __declspec(dllimport)
#   endif
#   define MRMESH_CLASS
#else
#   define MRMESH_API   __attribute__((visibility("default")))
// to fix undefined reference to `typeinfo/vtable
#   define MRMESH_CLASS __attribute__((visibility("default")))
#endif

namespace MR
{
 
class MRMESH_CLASS EdgeTag;
class MRMESH_CLASS UndirectedEdgeTag;
class MRMESH_CLASS FaceTag;
class MRMESH_CLASS VertTag;
class MRMESH_CLASS PixelTag;
class MRMESH_CLASS VoxelTag;

template <typename T> class Id;
template <typename T, typename I> class Vector;
template <typename T> class Buffer;

using EdgeId = Id<EdgeTag>;
using UndirectedEdgeId = Id<UndirectedEdgeTag>;
using FaceId = Id<FaceTag>;
using VertId = Id<VertTag>;
using PixelId = Id<PixelTag>;
using VoxelId = Id<VoxelTag>;
class ViewportId;
class ViewportMask;

/// three vertex ids describing a triangle topology
using ThreeVertIds = std::array<VertId, 3>;
/// mapping from FaceId to a triple of vertex indices
using Triangulation = Vector<ThreeVertIds, FaceId>;

using EdgePath = std::vector<EdgeId>;
using EdgeLoop = std::vector<EdgeId>;

class BitSet;
template <typename T> class TaggedBitSet;

struct Color;

using FaceBitSet = TaggedBitSet<FaceTag>;
using VertBitSet = TaggedBitSet<VertTag>;
using EdgeBitSet = TaggedBitSet<EdgeTag>;
using UndirectedEdgeBitSet = TaggedBitSet<UndirectedEdgeTag>;
using PixelBitSet = TaggedBitSet<PixelTag>;
using VoxelBitSet = TaggedBitSet<VoxelTag>;

template <typename T> class SetBitIteratorT;

using SetBitIterator     = SetBitIteratorT<BitSet>;
using FaceSetBitIterator = SetBitIteratorT<FaceBitSet>;
using VertSetBitIterator = SetBitIteratorT<VertBitSet>;
using EdgeSetBitIterator = SetBitIteratorT<EdgeBitSet>;
using UndirectedEdgeSetBitIterator = SetBitIteratorT<UndirectedEdgeBitSet>;

template <typename T> struct MRMESH_CLASS Vector2;
using Vector2b = Vector2<bool>;
using Vector2i = Vector2<int>;
using Vector2ll= Vector2<long long>;
using Vector2f = Vector2<float>;
using Vector2d = Vector2<double>;

template <typename T> struct MRMESH_CLASS Vector3;
using Vector3b = Vector3<bool>;
using Vector3i = Vector3<int>;
using Vector3ll= Vector3<long long>;
using Vector3f = Vector3<float>;
using Vector3d = Vector3<double>;

template <typename T> struct Vector4;
using Vector4b = Vector4<bool>;
using Vector4i = Vector4<int>;
using Vector4ll= Vector4<long long>;
using Vector4f = Vector4<float>;
using Vector4d = Vector4<double>;

template <typename T> struct Matrix2;
using Matrix2b = Matrix2<bool>;
using Matrix2i = Matrix2<int>;
using Matrix2ll= Matrix2<long long>;
using Matrix2f = Matrix2<float>;
using Matrix2d = Matrix2<double>;

template <typename T> struct Matrix3;
using Matrix3b = Matrix3<bool>;
using Matrix3i = Matrix3<int>;
using Matrix3ll= Matrix3<long long>;
using Matrix3f = Matrix3<float>;
using Matrix3d = Matrix3<double>;

template <typename T> struct Matrix4;
using Matrix4b = Matrix4<bool>;
using Matrix4i = Matrix4<int>;
using Matrix4ll= Matrix4<long long>;
using Matrix4f = Matrix4<float>;
using Matrix4d = Matrix4<double>;

template <typename T> struct SymMatrix2;
using SymMatrix2b = SymMatrix2<bool>;
using SymMatrix2i = SymMatrix2<int>;
using SymMatrix2ll= SymMatrix2<long long>;
using SymMatrix2f = SymMatrix2<float>;
using SymMatrix2d = SymMatrix2<double>;

template <typename T> struct SymMatrix3;
using SymMatrix3b = SymMatrix3<bool>;
using SymMatrix3i = SymMatrix3<int>;
using SymMatrix3ll= SymMatrix3<long long>;
using SymMatrix3f = SymMatrix3<float>;
using SymMatrix3d = SymMatrix3<double>;

template <typename V> struct AffineXf;
template <typename T> using AffineXf2 = AffineXf<Vector2<T>>;
using AffineXf2f = AffineXf2<float>;
using AffineXf2d = AffineXf2<double>;

template <typename T> using AffineXf3 = AffineXf<Vector3<T>>;
using AffineXf3f = AffineXf3<float>;
using AffineXf3d = AffineXf3<double>;

template <typename V> struct Line;
template <typename T> using Line2 = Line<Vector2<T>>;
using Line2f = Line2<float>;
using Line2d = Line2<double>;

template <typename T> using Line3 = Line<Vector3<T>>;
using Line3f = Line3<float>;
using Line3d = Line3<double>;

template <typename V> struct LineSegm;
template <typename T> using LineSegm2 = LineSegm<Vector2<T>>;
using LineSegm2f = LineSegm2<float>;
using LineSegm2d = LineSegm2<double>;

template <typename T> using LineSegm3 = LineSegm<Vector3<T>>;
using LineSegm3f = LineSegm3<float>;
using LineSegm3d = LineSegm3<double>;

template <typename V> using Contour = std::vector<V>;
template <typename T> using Contour2 = Contour<Vector2<T>>;
template <typename T> using Contour3 = Contour<Vector3<T>>;
using Contour2d = Contour2<double>;
using Contour2f = Contour2<float>;
using Contour3d = Contour3<double>;
using Contour3f = Contour3<float>;

template <typename V> using Contours = std::vector<Contour<V>>;
template <typename T> using Contours2 = Contours<Vector2<T>>;
template <typename T> using Contours3 = Contours<Vector3<T>>;
using Contours2d = Contours2<double>;
using Contours2f = Contours2<float>;
using Contours3d = Contours3<double>;
using Contours3f = Contours3<float>;

template <typename T> using Contour3 = Contour<Vector3<T>>;
using Contour3d = Contour3<double>;
using Contours3d = std::vector<Contour3d>;
using Contour3f = Contour3<float>;
using Contours3f = std::vector<Contour3f>;

template <typename T> struct Plane3;
using Plane3f = Plane3<float>;
using Plane3d = Plane3<double>;

template <typename V> struct Box;
template <typename T> using Box2 = Box<Vector2<T>>;
using Box2i = Box2<int>;
using Box2ll = Box2<long long>;
using Box2f = Box2<float>;
using Box2d = Box2<double>;

template <typename T> using Box3 = Box<Vector3<T>>;
using Box3i = Box3<int>;
using Box3ll = Box3<long long>;
using Box3f = Box3<float>;
using Box3d = Box3<double>;

template <typename V> struct QuadraticForm;

template <typename T> using QuadraticForm2 = QuadraticForm<Vector2<T>>;
using QuadraticForm2f = QuadraticForm2<float>;
using QuadraticForm2d = QuadraticForm2<double>;

template <typename T> using QuadraticForm3 = QuadraticForm<Vector3<T>>;
using QuadraticForm3f = QuadraticForm3<float>;
using QuadraticForm3d = QuadraticForm3<double>;

template <typename T> struct Quaternion;
using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

class PointAccumulator;

template <typename T> struct TriPoint;
using TriPointf = TriPoint<float>;
using TriPointd = TriPoint<double>;
struct PointOnFace;
struct MeshEdgePoint;
using SurfacePath = std::vector<MeshEdgePoint>;
struct MeshTriPoint;
struct MeshProjectionResult;
struct MeshIntersectionResult;
template <typename T> struct IntersectionPrecomputes;

template <typename I> struct IteratorRange;

using FaceMap = Vector<FaceId, FaceId>;
using VertMap = Vector<VertId, VertId>;
using EdgeMap = Vector<EdgeId, EdgeId>;
using UndirectedEdgeMap = Vector<UndirectedEdgeId, UndirectedEdgeId>;
///  mapping of whole edges: map[e]->f, map[e.sym()]->f.sym(), where only map[e] for even edges is stored
using WholeEdgeMap = Vector<EdgeId, UndirectedEdgeId>;
using VertCoords = Vector<Vector3f, VertId>;
using VertNormals = Vector<Vector3f, VertId>;
using FaceNormals = Vector<Vector3f, FaceId>;

template <typename K>
using HashSet = phmap::flat_hash_set<K>;
template <typename K>
using ParallelHashSet = phmap::parallel_flat_hash_set<K>;

using FaceHashSet = HashSet<FaceId>;
using VertHashSet = HashSet<VertId>;
using EdgeHashSet = HashSet<EdgeId>;

template <typename K, typename V>
using HashMap = phmap::flat_hash_map<K, V>;
template <typename K, typename V>
using ParallelHashMap = phmap::parallel_flat_hash_map<K, V>;

using FaceHashMap = HashMap<FaceId, FaceId>;
using VertHashMap = HashMap<VertId, VertId>;
using EdgeHashMap = HashMap<EdgeId, EdgeId>;
using UndirectedEdgeHashMap = HashMap<UndirectedEdgeId, UndirectedEdgeId>;
///  mapping of whole edges: map[e]->f, map[e.sym()]->f.sym(), where only map[e] for even edges is stored
using WholeEdgeHashMap = HashMap<UndirectedEdgeId, EdgeId>;

template <typename I> class UnionFind;
template <typename T, typename I, typename P> class Heap;

class MeshTopology;
struct Mesh;
struct MeshPart;
struct PointCloud;
class MRMESH_CLASS AABBTree;
class MRMESH_CLASS AABBTreePoints;
template<typename T> class UniqueThreadSafeOwner;

class PolylineTopology;
template<typename V>
struct Polyline;
using Polyline2 = Polyline<Vector2f>;
using Polyline3 = Polyline<Vector3f>;
template<typename V>
class MRMESH_CLASS AABBTreePolyline;
using AABBTreePolyline2 = AABBTreePolyline<Vector2f>;
using AABBTreePolyline3 = AABBTreePolyline<Vector3f>;

template<typename V> struct [[nodiscard]] PolylineProjectionResult;
using PolylineProjectionResult2 = PolylineProjectionResult<Vector2f>;
using PolylineProjectionResult3 = PolylineProjectionResult<Vector3f>;

template<typename V> struct [[nodiscard]] PolylineProjectionWithOffsetResult;
using Polyline2ProjectionWithOffsetResult = PolylineProjectionWithOffsetResult<Vector2f>;
using PolylineProjectionWithOffsetResult3 = PolylineProjectionWithOffsetResult<Vector3f>;

class DistanceMap;

class Object;
class VisualObject;
class ObjectMeshHolder;
class ObjectMesh;
class ObjectPointsHolder;
class ObjectPoints;
class ObjectLinesHolder;
class ObjectLines;
class ObjectDistanceMap;
class ObjectLabel;
class PointObject;
class LineObject;
class CircleObject;
class PlaneObject;
class SphereObject;

template <typename T>
struct VoxelsVolume;
using SimpleVolume = VoxelsVolume<std::vector<float>>;

#ifndef MRMESH_NO_VOXEL
class ObjectVoxels;

struct OpenVdbFloatGrid;
using FloatGrid = std::shared_ptr<OpenVdbFloatGrid>;
using VdbVolume = VoxelsVolume<FloatGrid>;
#endif

class HistoryAction;
class MRMESH_CLASS HistoryStore;
class ChangeObjectAction;
class MRMESH_CLASS ChangeSceneAction;
class ChangeMeshFaceSelectionAction;
class ChangeMeshEdgeSelectionAction;
class ChangeMeshCreasesAction;
class ChangePointPointSelectionAction;
class ChangeMeshAction;
class ChangeMeshPointsAction;
class ChangeMeshTopologyAction;
class ChangeXfAction;
class CombinedHistoryAction;
class SwapRootAction;

template <typename Tag>
class MRMESH_CLASS ColorMapAggregator;
using VertColorMapAggregator = ColorMapAggregator<VertTag>;
using UndirEdgeColorMapAggregator = ColorMapAggregator<UndirectedEdgeTag>;
using FaceColorMapAggregator = ColorMapAggregator<FaceTag>;

enum class FilterType : char
{
    Linear,
    Discrete
};

enum class WrapType : char
{
    Repeat,
    Mirror,
    Clamp
};

template <typename T>
constexpr inline T sqr( T x ) noexcept { return x * x; }

template <typename T>
constexpr inline int sgn( T x ) noexcept { return x > 0 ? 1 : ( x < 0 ? -1 : 0 ); }

template<typename...> 
inline constexpr bool dependent_false = false;

namespace MeshBuilder
{

struct BuildSettings;
struct Triangle;
struct VertDuplication;

} //namespace MeshBuilder

} //namespace MR
