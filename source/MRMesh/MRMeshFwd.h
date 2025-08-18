#pragma once

#include "config.h"

#include "MRMesh/MRCanonicalTypedefs.h"
#include "MRPch/MRBindingMacros.h"

// Not-zero _ITERATOR_DEBUG_LEVEL in Microsoft STL greatly reduces the performance of STL containers.
//
// Pre-build binaries from MeshLib distribution are prepared with _ITERATOR_DEBUG_LEVEL=0,
// and if you build MeshLib by yourself then _ITERATOR_DEBUG_LEVEL=0 is also selected see
// 1) vcpkg/triplets/x64-windows-meshlib.cmake and
// 2) MeshLib/source/common.props
// Please note that all other modules (.exe, .dll, .lib) with MS STL calls in your application also need
// to define exactly the same value of _ITERATOR_DEBUG_LEVEL to be operational after linking.
//
// If you deliberately would like to work with not zero _ITERATOR_DEBUG_LEVEL, then please define
// additionally MR_ITERATOR_DEBUG_LEVEL with the same value to indicate that it is done intentionally
// (and you are ok with up to 100x slowdown).
//
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

// Check C++ version.
#ifdef _MSC_VER
#define MR_CPP_STANDARD_DATE _MSVC_LANG
#else
#define MR_CPP_STANDARD_DATE __cplusplus
#endif
// Note `201709`. C++20 usually sets `202002`, while C++17 sets `201703`.
//   This `201709` is what GCC 10 sets on `-std=c++20` (presumably to indicate incomplete implementation?).
//   Other compilers we use don't have this issue.
// Also note `__CUDACC__` - currently our Cuda code is compiled as C++17 (compiler doesn't support C++20?),
//   and we carefully avoid headers incomaptible with C++17.
#if MR_CPP_STANDARD_DATE < 201709 && !defined(__CUDACC__)
#error Must enable C++20 or newer!
#endif
// -- Curently we have no macros that don't work with the old preprocessor. Leaving the check here to possibly be enabled later.
// Reject old MSVC preprocessor.
// Note that we exclude Cuda here. Not 100% sure if it has a good preprocessor, or we just avoid including the headers sensitive to it in Cuda.
// #if defined(_MSC_VER) && !defined(__clang__) && (!defined(_MSVC_TRADITIONAL) || _MSVC_TRADITIONAL == 1) && !defined(__CUDACC__)
// #error MSVC users must enable the new standard-conformant preprocessor using `/Zc:preprocessor`!
// #endif


#if defined(__GNUC__) && (__GNUC__ >= 13 && __GNUC__ <= 15)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Warray-bounds"
  #pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

#include <array>

#if defined(__GNUC__) && (__GNUC__ >= 13 && __GNUC__ <= 15)
  #pragma GCC diagnostic pop
#endif

#include <vector>
#include <string>
#include <parallel_hashmap/phmap_fwd_decl.h>
#include <functional>

#ifdef _WIN32
#   ifdef MRMesh_EXPORTS
#       define MRMESH_API __declspec(dllexport)
#   else
#       define MRMESH_API __declspec(dllimport)
#   endif
#   define MRMESH_CLASS
#else
#   define MRMESH_API   __attribute__((visibility("default")))
// to fix undefined reference to `typeinfo/vtable`
// Also it's important to use this on any type for which `typeid` is used in multiple shared libraries, and then passed across library boundaries.
//   Otherwise on Mac the resulting typeids will incorrectly compare not equal.
#   define MRMESH_CLASS __attribute__((visibility("default")))
#endif

namespace MR
{

struct NoInit {};
inline constexpr NoInit noInit;
template <typename T> struct MRMESH_CLASS NoDefInit;

class MRMESH_CLASS EdgeTag;
class MRMESH_CLASS UndirectedEdgeTag;
class MRMESH_CLASS FaceTag;
class MRMESH_CLASS VertTag;
class MRMESH_CLASS PixelTag;
class MRMESH_CLASS VoxelTag;
class MRMESH_CLASS RegionTag;
class MRMESH_CLASS NodeTag;
class MRMESH_CLASS ObjTag;
class MRMESH_CLASS TextureTag;
class MRMESH_CLASS GraphVertTag;
class MRMESH_CLASS GraphEdgeTag;

MR_CANONICAL_TYPEDEFS( (template <typename T> class MRMESH_CLASS), Id,
    ( EdgeId,           Id<EdgeTag>           )
    ( UndirectedEdgeId, Id<UndirectedEdgeTag> )
    ( FaceId,           Id<FaceTag>           )
    ( VertId,           Id<VertTag>           )
    ( PixelId,          Id<PixelTag>          )
    ( VoxelId,          Id<VoxelTag>          )
    ( RegionId,         Id<RegionTag>         )
    ( NodeId,           Id<NodeTag>           )
    ( ObjId,            Id<ObjTag>            )
    ( TextureId,        Id<TextureTag>        )
    ( GraphVertId,      Id<GraphVertTag>      )
    ( GraphEdgeId,      Id<GraphEdgeTag>      )
)
// Those are full specializations in `MRId.h`, so `MR_CANONICAL_TYPEDEFS` doesn't work on them.
// Have to add this too.
template <> class MR_BIND_PREFERRED_NAME(MR::EdgeId) Id<EdgeTag>;
template <> class MR_BIND_PREFERRED_NAME(MR::VoxelId) Id<VoxelTag>;

MR_CANONICAL_TYPEDEFS( (template <typename T> class MRMESH_CLASS), NoInitId,
    ( NoInitNodeId, NoInitId<NodeTag> )
)

template <typename T, typename I = size_t> class MRMESH_CLASS Buffer;
struct PackMapping;

class ViewportId;
class ViewportMask;

struct UnorientedTriangle;
struct SomeLocalTriangulations;
struct AllLocalTriangulations;

using EdgePath = std::vector<EdgeId>;
using EdgeLoop = std::vector<EdgeId>;
using EdgeLoops = std::vector<EdgeLoop>;

class MRMESH_CLASS BitSet;

MR_CANONICAL_TYPEDEFS( (template <typename I> class MRMESH_CLASS), TypedBitSet,
    ( FaceBitSet,           TypedBitSet<FaceId>           )
    ( VertBitSet,           TypedBitSet<VertId>           )
    ( EdgeBitSet,           TypedBitSet<EdgeId>           )
    ( UndirectedEdgeBitSet, TypedBitSet<UndirectedEdgeId> )
    ( PixelBitSet,          TypedBitSet<PixelId>          )
    ( VoxelBitSet,          TypedBitSet<VoxelId>          )
    ( RegionBitSet,         TypedBitSet<RegionId>         )
    ( NodeBitSet,           TypedBitSet<NodeId>           )
    ( ObjBitSet,            TypedBitSet<ObjId>            )
    ( TextureBitSet,        TypedBitSet<TextureId>        )
    ( GraphVertBitSet,      TypedBitSet<GraphVertId>      )
    ( GraphEdgeBitSet,      TypedBitSet<GraphEdgeId>      )
)

template<typename T>
using TaggedBitSet = TypedBitSet<Id<T>>;

MR_CANONICAL_TYPEDEFS( (template <typename T> class MRMESH_CLASS), SetBitIteratorT,
    ( SetBitIterator,               SetBitIteratorT<BitSet>               )
    ( FaceSetBitIterator,           SetBitIteratorT<FaceBitSet>           )
    ( VertSetBitIterator,           SetBitIteratorT<VertBitSet>           )
    ( EdgeSetBitIterator,           SetBitIteratorT<EdgeBitSet>           )
    ( UndirectedEdgeSetBitIterator, SetBitIteratorT<UndirectedEdgeBitSet> )
)

struct Color;

// Must use our `Int64` and `Uint64` everywhere in the public API instead of `long`, `long long`, `int64_t`, `uint64_t`, etc.
// But `size_t` and `ptrdiff_t` are allowed as is.
// This is required to generate consistent C bindings on all platforms, and this is checked during binding generation.
//
// Those standard typedefs are a mess across platforms. On Windows, they all expand to `long long`, because `long` is not wide enough.
// On Linux, they all expand to `long`, which again makes sense, because it's wide enough, and it's natural to use it instead of `long long` if it's wide enough.
// This difference alone causes issues, but what's worse is that on Mac, the typedefs use both `long` and `long long`, despite them having
//   the same size (anything with digits in the name is `long long`, everything else is `long`, for some reason).
// The only way around that is to only use `long` or `long long` in the interface per platform, but not both. Then during binding generation,
//   replace that type with `int64_t`, and complain if we got another one. This only makes sense on 64-bit platforms, but we only generate bindings on those.
// Since `size_t` and `ptrdiff_t` are so important, we allow those, and anything that expands to the same type. Sadly `[u]int64_t` doesn't on Mac,
//   so we can't use it directly, and have to instead use our own typedefs defined below. The untypedefed `long` and `long long` can't be used either,
//   for the same reasons (whether they match `size_t` depends on the platform, and we can't allow that).
#ifdef __APPLE__
using Int64 = std::ptrdiff_t;
using Uint64 = std::size_t;
static_assert(sizeof(Int64) == 8);
static_assert(sizeof(Uint64) == 8);
#else
using Int64 = std::int64_t;
using Uint64 = std::uint64_t;
#endif

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), MRMESH_CLASS Vector2,
    ( Vector2b,   Vector2<bool>   )
    ( Vector2i,   Vector2<int>    )
    ( Vector2i64, Vector2<Int64>  )
    ( Vector2f,   Vector2<float>  )
    ( Vector2d,   Vector2<double> )
)
// See `Int64` above for why this is deprecated.
// This is behind an `#ifdef` to avoid instantiating `Vector2<T>` with `T == long long`, which happens even if the typedef has `MR_BIND_IGNORE`,
//   which is a bug.
#if !MR_PARSING_FOR_ANY_BINDINGS
using Vector2ll [[deprecated("Use `Vector2i64` instead.")]] = Vector2<long long>;
#endif

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), MRMESH_CLASS Vector3,
    ( Vector3b,   Vector3<bool>   )
    ( Vector3i,   Vector3<int>    )
    ( Vector3i64, Vector3<Int64>  )
    ( Vector3f,   Vector3<float>  )
    ( Vector3d,   Vector3<double> )
)
#if !MR_PARSING_FOR_ANY_BINDINGS
using Vector3ll [[deprecated("Use `Vector3i64` instead.")]] = Vector3<long long>;
#endif

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), Vector4,
    ( Vector4b,   Vector4<bool>   )
    ( Vector4i,   Vector4<int>    )
    ( Vector4i64, Vector4<Int64>  )
    ( Vector4f,   Vector4<float>  )
    ( Vector4d,   Vector4<double> )
)
#if !MR_PARSING_FOR_ANY_BINDINGS
using Vector4ll [[deprecated("Use `Vector4i64` instead.")]] = Vector4<long long>;
#endif

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), Matrix2,
    ( Matrix2b,   Matrix2<bool>   )
    ( Matrix2i,   Matrix2<int>    )
    ( Matrix2i64, Matrix2<Int64>  )
    ( Matrix2f,   Matrix2<float>  )
    ( Matrix2d,   Matrix2<double> )
)
#if !MR_PARSING_FOR_ANY_BINDINGS
using Matrix2ll [[deprecated("Use `Matrix2i64` instead.")]] = Matrix2<long long>;
#endif

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), Matrix3,
    ( Matrix3b,   Matrix3<bool>   )
    ( Matrix3i,   Matrix3<int>    )
    ( Matrix3i64, Matrix3<Int64>  )
    ( Matrix3f,   Matrix3<float>  )
    ( Matrix3d,   Matrix3<double> )
)
#if !MR_PARSING_FOR_ANY_BINDINGS
using Matrix3ll [[deprecated("Use `Matrix3i64` instead.")]] = Matrix3<long long>;
#endif

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), Matrix4,
    ( Matrix4b,   Matrix4<bool>   )
    ( Matrix4i,   Matrix4<int>    )
    ( Matrix4i64, Matrix4<Int64>  )
    ( Matrix4f,   Matrix4<float>  )
    ( Matrix4d,   Matrix4<double> )
)
#if !MR_PARSING_FOR_ANY_BINDINGS
using Matrix4ll [[deprecated("Use `Matrix4i64` instead.")]] = Matrix4<long long>;
#endif

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), SymMatrix2,
    ( SymMatrix2b,   SymMatrix2<bool>   )
    ( SymMatrix2i,   SymMatrix2<int>    )
    ( SymMatrix2i64, SymMatrix2<Int64>  )
    ( SymMatrix2f,   SymMatrix2<float>  )
    ( SymMatrix2d,   SymMatrix2<double> )
)
#if !MR_PARSING_FOR_ANY_BINDINGS
using SymMatrix2ll [[deprecated("Use `SymMatrix2i64` instead.")]] = SymMatrix2<long long>;
#endif

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), SymMatrix3,
    ( SymMatrix3b,   SymMatrix3<bool>   )
    ( SymMatrix3i,   SymMatrix3<int>    )
    ( SymMatrix3i64, SymMatrix3<Int64>  )
    ( SymMatrix3f,   SymMatrix3<float>  )
    ( SymMatrix3d,   SymMatrix3<double> )
)
#if !MR_PARSING_FOR_ANY_BINDINGS
using SymMatrix3ll [[deprecated("Use `SymMatrix3i64` instead.")]] = SymMatrix3<long long>;
#endif

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), SymMatrix4,
    ( SymMatrix4b,   SymMatrix4<bool>   )
    ( SymMatrix4i,   SymMatrix4<int>    )
    ( SymMatrix4i64, SymMatrix4<Int64>  )
    ( SymMatrix4f,   SymMatrix4<float>  )
    ( SymMatrix4d,   SymMatrix4<double> )
)
#if !MR_PARSING_FOR_ANY_BINDINGS
using SymMatrix4ll [[deprecated("Use `SymMatrix4i64` instead.")]] = SymMatrix4<long long>;
#endif

MR_CANONICAL_TYPEDEFS( (template <typename V> struct), AffineXf,
    ( AffineXf2f, AffineXf<Vector2<float>>  )
    ( AffineXf2d, AffineXf<Vector2<double>> )
    ( AffineXf3f, AffineXf<Vector3<float>>  )
    ( AffineXf3d, AffineXf<Vector3<double>> )
)
template <typename T> using AffineXf2 = AffineXf<Vector2<T>>;
template <typename T> using AffineXf3 = AffineXf<Vector3<T>>;

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), RigidXf3,
    ( RigidXf3f, RigidXf3<float>  )
    ( RigidXf3d, RigidXf3<double> )
)

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), RigidScaleXf3,
    ( RigidScaleXf3f, RigidScaleXf3<float>  )
    ( RigidScaleXf3d, RigidScaleXf3<double> )
)

class PointToPointAligningTransform;
class PointToPlaneAligningTransform;

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), Sphere,
    ( Sphere2f, Sphere<Vector2<float>>  )
    ( Sphere2d, Sphere<Vector2<double>> )
    ( Sphere3f, Sphere<Vector3<float>>  )
    ( Sphere3d, Sphere<Vector3<double>> )
)
template <typename T> using Sphere2 = Sphere<Vector2<T>>;
template <typename T> using Sphere3 = Sphere<Vector3<T>>;

MR_CANONICAL_TYPEDEFS( (template <typename V> struct), Line,
    ( Line2f, Line<Vector2<float>>  )
    ( Line2d, Line<Vector2<double>> )
    ( Line3f, Line<Vector3<float>>  )
    ( Line3d, Line<Vector3<double>> )
)
template <typename T> using Line2 = Line<Vector2<T>>;
template <typename T> using Line3 = Line<Vector3<T>>;

MR_CANONICAL_TYPEDEFS( (template <typename V> struct), LineSegm,
    ( LineSegm2f, LineSegm<Vector2<float>>  )
    ( LineSegm2d, LineSegm<Vector2<double>> )
    ( LineSegm3f, LineSegm<Vector3<float>>  )
    ( LineSegm3d, LineSegm<Vector3<double>> )
)
template <typename T> using LineSegm2 = LineSegm<Vector2<T>>;
template <typename T> using LineSegm3 = LineSegm<Vector3<T>>;

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), Parabola,
    ( Parabolaf, Parabola<float>  )
    ( Parabolad, Parabola<double> )
)

MR_CANONICAL_TYPEDEFS( (template <typename T> class), BestFitParabola,
    ( BestFitParabolaf, BestFitParabola<float>  )
    ( BestFitParabolad, BestFitParabola<double> )
)

MR_CANONICAL_TYPEDEFS( (template <typename T> class), Cylinder3,
    ( Cylinder3f, Cylinder3<float>  )
    ( Cylinder3d, Cylinder3<double> )
)

MR_CANONICAL_TYPEDEFS( (template <typename T> class), Cone3,
    ( Cone3f, Cone3<float>  )
    ( Cone3d, Cone3<double> )
)

// No canonical typedefs here, because those ultimately boil to `std::vector`, which isn't under our control.
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

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), Plane3,
    ( Plane3f, Plane3<float>  )
    ( Plane3d, Plane3<double> )
)

MR_CANONICAL_TYPEDEFS( (template <typename V> struct MRMESH_CLASS), Box,
    ( Box1i,   Box<int>             )
    ( Box1i64, Box<Int64>           )
    ( Box1f,   Box<float>           )
    ( Box1d,   Box<double>          )
    ( Box2i,   Box<Vector2<int>>    )
    ( Box2i64, Box<Vector2<Int64>>  )
    ( Box2f,   Box<Vector2<float>>  )
    ( Box2d,   Box<Vector2<double>> )
    ( Box3i,   Box<Vector3<int>>    )
    ( Box3i64, Box<Vector3<Int64>>  )
    ( Box3f,   Box<Vector3<float>>  )
    ( Box3d,   Box<Vector3<double>> )
)
#if !MR_PARSING_FOR_ANY_BINDINGS
using Box1ll [[deprecated("Use `Box1i64` instead.")]] = Box<long long>;
using Box2ll [[deprecated("Use `Box2i64` instead.")]] = Box<Vector2<long long>>;
using Box3ll [[deprecated("Use `Box3i64` instead.")]] = Box<Vector3<long long>>;
#endif

template <typename T> using MinMax = Box<T>;
using MinMaxf = MinMax<float>;
using MinMaxd = MinMax<double>;
using MinMaxi = MinMax<int>;

template <typename T> using Box1 = Box<T>;
template <typename T> using Box2 = Box<Vector2<T>>;
template <typename T> using Box3 = Box<Vector3<T>>;

template<typename T, typename I> struct MinArg;
template<typename T, typename I> struct MaxArg;
template<typename T, typename I> struct MinMaxArg;

MR_CANONICAL_TYPEDEFS( (template <typename V> struct MRMESH_CLASS), Ball,
    ( Ball1f,  Ball<float>              )
    ( Ball1d,  Ball<double>             )
    ( Ball2f,  Ball<Vector2<float>>     )
    ( Ball2d,  Ball<Vector2<double>>    )
    ( Ball3f,  Ball<Vector3<float>>     )
    ( Ball3d,  Ball<Vector3<double>>    )
)
template <typename T> using Ball1 = Ball<T>;
template <typename T> using Ball2 = Ball<Vector2<T>>;
template <typename T> using Ball3 = Ball<Vector3<T>>;

MR_CANONICAL_TYPEDEFS( (template <typename V> struct MRMESH_CLASS), CubicBezierCurve,
    ( CubicBezierCurve2f,  CubicBezierCurve<Vector2<float>>     )
    ( CubicBezierCurve2d,  CubicBezierCurve<Vector2<double>>    )
    ( CubicBezierCurve3f,  CubicBezierCurve<Vector3<float>>     )
    ( CubicBezierCurve3d,  CubicBezierCurve<Vector3<double>>    )
)
template <typename T> using CubicBezierCurve2 = CubicBezierCurve<Vector2<T>>;
template <typename T> using CubicBezierCurve3 = CubicBezierCurve<Vector3<T>>;

MR_CANONICAL_TYPEDEFS( (template <typename V> struct), QuadraticForm,
    ( QuadraticForm2f, QuadraticForm<Vector2<float>>  )
    ( QuadraticForm2d, QuadraticForm<Vector2<double>> )
    ( QuadraticForm3f, QuadraticForm<Vector3<float>>  )
    ( QuadraticForm3d, QuadraticForm<Vector3<double>> )
)
template <typename T> using QuadraticForm2 = QuadraticForm<Vector2<T>>;
template <typename T> using QuadraticForm3 = QuadraticForm<Vector3<T>>;

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), Quaternion,
    ( Quaternionf, Quaternion<float>  )
    ( Quaterniond, Quaternion<double> )
)

// No canonical typedefs because `std::array` is not under our control.
template <typename T> using Triangle3 = std::array<Vector3<T>, 3>;
using Triangle3i = Triangle3<int>;
using Triangle3f = Triangle3<float>;
using Triangle3d = Triangle3<double>;

class PointAccumulator;

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), SegmPoint,
    ( SegmPointf, SegmPoint<float>  )
    ( SegmPointd, SegmPoint<double> )
)

struct EdgePoint;
struct EdgeSegment;
using MeshEdgePoint = EdgePoint;
using SurfacePath = std::vector<MeshEdgePoint>;
using SurfacePaths = std::vector<SurfacePath>;
using IsoLine = SurfacePath;
using IsoLines = SurfacePaths;
using PlaneSection = SurfacePath;
using PlaneSections = SurfacePaths;
struct EdgePointPair;
class Laplacian;

using VertPair = std::pair<VertId, VertId>;
using FacePair = std::pair<FaceId, FaceId>;
using EdgePair = std::pair<EdgeId, EdgeId>;
using UndirectedEdgePair = std::pair<UndirectedEdgeId, UndirectedEdgeId>;

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), TriPoint,
    ( TriPointf, TriPoint<float>  )
    ( TriPointd, TriPoint<double> )
)

struct PointOnFace;
struct PointOnObject;
struct MeshTriPoint;
struct MeshProjectionResult;
struct MeshIntersectionResult;
struct PointsProjectionResult;
template <typename T> struct IntersectionPrecomputes;

template <typename I> struct IteratorRange;

/// Coordinates on texture
/// \param x,y should be in range [0..1], otherwise result depends on wrap type of texture (no need to clamp it, it is done on GPU if wrap type is "Clamp" )
using UVCoord = Vector2f;

/// two vertex ids describing an edge with the ends in vertices given by their ids
using TwoVertIds = std::array<VertId, 2>;

/// three vertex ids describing a triangle with the corners in vertices given by their ids
using ThreeVertIds = std::array<VertId, 3>;

struct MRMESH_CLASS Dipole;

MR_CANONICAL_TYPEDEFS( (template <typename T, typename I> class MRMESH_CLASS), Vector,
    /// mapping from UndirectedEdgeId to its end vertices
    ( Edges,    Vector<TwoVertIds, UndirectedEdgeId> )

    /// mapping from FaceId to a triple of vertex indices
    ( Triangulation,  Vector<ThreeVertIds, FaceId> )

    ( Dipoles,  Vector<Dipole, NodeId> )

    ( FaceMap,  Vector<FaceId, FaceId> )
    ( VertMap,  Vector<VertId, VertId> )
    ( EdgeMap,  Vector<EdgeId, EdgeId> )
    ( UndirectedEdgeMap,  Vector<UndirectedEdgeId, UndirectedEdgeId> )
    ( ObjMap,  Vector<ObjId, ObjId> )

    ///  mapping of whole edges: map[e]->f, map[e.sym()]->f.sym(), where only map[e] for even edges is stored
    ( WholeEdgeMap,  Vector<EdgeId, UndirectedEdgeId> )
    ( UndirectedEdge2RegionMap,  Vector<RegionId, UndirectedEdgeId> )
    ( Face2RegionMap,  Vector<RegionId, FaceId> )
    ( Vert2RegionMap,  Vector<RegionId, VertId> )

    ( VertCoords,  Vector<Vector3f, VertId> )
    ( VertCoords2, Vector<Vector2f, VertId> )
    ( VertNormals,  Vector<Vector3f, VertId> )
    ( VertUVCoords,  Vector<UVCoord, VertId> )
    ( FaceNormals,  Vector<Vector3f, FaceId> )

    ( TexturePerFace,  Vector<TextureId, FaceId> )
    ( VertColors,  Vector<Color, VertId> )
    ( FaceColors,  Vector<Color, FaceId> )
    ( EdgeColors,  Vector<Color, EdgeId> )
    ( UndirectedEdgeColors,  Vector<Color, UndirectedEdgeId> )

    ( VertScalars,  Vector<float, VertId> )
    ( FaceScalars,  Vector<float, FaceId> )
    ( EdgeScalars,  Vector<float, EdgeId> )
    ( UndirectedEdgeScalars,  Vector<float, UndirectedEdgeId> )
)

using VertPredicate = std::function<bool( VertId )>;
using FacePredicate = std::function<bool( FaceId )>;
using EdgePredicate = std::function<bool( EdgeId )>;
using UndirectedEdgePredicate = std::function<bool( UndirectedEdgeId )>;

using PreCollapseCallback = std::function<bool( EdgeId edgeToCollapse, const Vector3f& newEdgeOrgPos )>;
using OnEdgeSplit = std::function<void( EdgeId e1, EdgeId e )>;

template <typename T>
[[nodiscard]] inline bool contains( const std::function<bool( Id<T> )> & pred, Id<T> id )
{
    return id.valid() && ( !pred || pred( id ) );
}

using VertMetric = std::function<float( VertId )>;
using FaceMetric = std::function<float( FaceId )>;
using EdgeMetric = std::function<float( EdgeId )>;
using UndirectedEdgeMetric = std::function<float( UndirectedEdgeId )>;

MR_CANONICAL_TYPEDEFS( (template <typename T, typename I> struct MRMESH_CLASS), BMap,
    ( FaceBMap, BMap<FaceId, FaceId> )
    ( VertBMap, BMap<VertId, VertId> )
    ( EdgeBMap, BMap<EdgeId, EdgeId> )
    ( UndirectedEdgeBMap, BMap<UndirectedEdgeId, UndirectedEdgeId> )
    ( WholeEdgeBMap, BMap<EdgeId, UndirectedEdgeId> )
)

template <typename T, typename Hash = phmap::priv::hash_default_hash<T>, typename Eq = phmap::priv::hash_default_eq<T>>
using HashSet = phmap::flat_hash_set<T, Hash, Eq>;
template <typename T, typename Hash = phmap::priv::hash_default_hash<T>, typename Eq = phmap::priv::hash_default_eq<T>>
using ParallelHashSet = phmap::parallel_flat_hash_set<T, Hash, Eq>;

// No canonical typedefs because `phmap::...` is not under our control.
using FaceHashSet = HashSet<FaceId>;
using VertHashSet = HashSet<VertId>;
using EdgeHashSet = HashSet<EdgeId>;

template <typename K, typename V, typename Hash = phmap::priv::hash_default_hash<K>, typename Eq = phmap::priv::hash_default_eq<K>>
using HashMap = phmap::flat_hash_map<K, V, Hash, Eq>;
template <typename K, typename V, typename Hash = phmap::priv::hash_default_hash<K>, typename Eq = phmap::priv::hash_default_eq<K>>
using ParallelHashMap = phmap::parallel_flat_hash_map<K, V, Hash, Eq>;

using FaceHashMap = HashMap<FaceId, FaceId>;
using VertHashMap = HashMap<VertId, VertId>;
using EdgeHashMap = HashMap<EdgeId, EdgeId>;
using UndirectedEdgeHashMap = HashMap<UndirectedEdgeId, UndirectedEdgeId>;
///  mapping of whole edges: map[e]->f, map[e.sym()]->f.sym(), where only map[e] for even edges is stored
using WholeEdgeHashMap = HashMap<UndirectedEdgeId, EdgeId>;

template <typename K, typename V>
struct MapOrHashMap;

using FaceMapOrHashMap = MapOrHashMap<FaceId, FaceId>;
using VertMapOrHashMap = MapOrHashMap<VertId, VertId>;
using EdgeMapOrHashMap = MapOrHashMap<EdgeId, EdgeId>;
using UndirectedEdgeMapOrHashMap = MapOrHashMap<UndirectedEdgeId, UndirectedEdgeId>;
///  mapping of whole edges: map[e]->f, map[e.sym()]->f.sym(), where only map[e] for even edges is stored
using WholeEdgeMapOrHashMap = MapOrHashMap<UndirectedEdgeId, EdgeId>;

template <typename I> class UnionFind;
template <typename T, typename I, typename P> class Heap;

class MRMESH_CLASS MeshTopology;
struct MRMESH_CLASS Mesh;
struct MRMESH_CLASS EdgeLengthMesh;
class MRMESH_CLASS MeshOrPoints;
struct MRMESH_CLASS PointCloud;
struct MRMESH_CLASS PointCloudPart;
class MRMESH_CLASS AABBTree;
class MRMESH_CLASS AABBTreePoints;
class MRMESH_CLASS AABBTreeObjects;
struct MRMESH_CLASS CloudPartMapping;
struct MRMESH_CLASS PartMapping;
struct MeshOrPointsXf;
struct MeshTexture;
struct GridSettings;
struct TriMesh;

MR_CANONICAL_TYPEDEFS( ( template <typename T> struct ), MRMESH_CLASS MeshRegion,
    ( MeshPart, MeshRegion<FaceTag> )
    ( MeshVertPart, MeshRegion<VertTag> )
)

template<typename T> class UniqueThreadSafeOwner;
template<typename T> class SharedThreadSafeOwner;

class PolylineTopology;

MR_CANONICAL_TYPEDEFS( (template<typename V> struct), Polyline,
    ( Polyline2, Polyline<Vector2f> )
    ( Polyline3, Polyline<Vector3f> )
)

MR_CANONICAL_TYPEDEFS( (template <typename V> class MRMESH_CLASS), AABBTreePolyline,
    ( AABBTreePolyline2, AABBTreePolyline<Vector2f> )
    ( AABBTreePolyline3, AABBTreePolyline<Vector3f> )
)

template<typename T> struct IntersectionPrecomputes;
template<typename T> struct IntersectionPrecomputes2;

MR_CANONICAL_TYPEDEFS( (template<typename V> struct [[nodiscard]]), PolylineProjectionResult,
    ( PolylineProjectionResult2, PolylineProjectionResult<Vector2f> )
    ( PolylineProjectionResult3, PolylineProjectionResult<Vector3f> )
)

MR_CANONICAL_TYPEDEFS( (template<typename V> struct [[nodiscard]]), PolylineProjectionWithOffsetResult,
    ( Polyline2ProjectionWithOffsetResult, PolylineProjectionWithOffsetResult<Vector2f> )
    ( PolylineProjectionWithOffsetResult3, PolylineProjectionWithOffsetResult<Vector3f> )
)

class DistanceMap;
struct DistanceToMeshOptions;
struct SignedDistanceToMeshOptions;

using GcodeSource = std::vector<std::string>;

class Object;
class SceneRootObject;
class VisualObject;
class ObjectMeshHolder;
class ObjectMesh;
struct ObjectMeshData;
class ObjectPointsHolder;
class ObjectPoints;
class ObjectLinesHolder;
class ObjectLines;
class ObjectDistanceMap;
class ObjectLabel;
class ObjectGcode;
class PointObject;
class LineObject;
class CircleObject;
class PlaneObject;
class SphereObject;
class CylinderObject;
class ConeObject;

struct LoadedObjects;

struct Image;
class AnyVisualizeMaskEnum;

class HistoryAction;
class ChangeObjectAction;
class MRMESH_CLASS ChangeSceneAction;
class ChangeMeshFaceSelectionAction;
class ChangeMeshEdgeSelectionAction;
class ChangeMeshCreasesAction;
class ChangePointPointSelectionAction;
class ChangeMeshAction;
class ChangeMeshDataAction;
class ChangeMeshPointsAction;
class ChangeMeshTopologyAction;
class ChangeXfAction;
class CombinedHistoryAction;
class SwapRootAction;

MR_CANONICAL_TYPEDEFS( (template <typename Tag> class MRMESH_CLASS), ColorMapAggregator,
    ( VertColorMapAggregator, ColorMapAggregator<VertTag> )
    ( UndirEdgeColorMapAggregator, ColorMapAggregator<UndirectedEdgeTag> )
    ( FaceColorMapAggregator, ColorMapAggregator<FaceTag> )
)

template<typename T>
class FewSmallest;

class Graph;
class WatershedGraph;

struct TbbTaskArenaAndGroup;

/// Argument value - progress in [0,1];
/// returns true to continue the operation and returns false to stop the operation
/// \ingroup BasicStructuresGroup
typedef std::function<bool( float )> ProgressCallback;

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

/// determines how points to be ordered
enum class Reorder : char
{
    None,              ///< the order is not changed
    Lexicographically, ///< the order is determined by lexicographical sorting by coordinates (optimal for uniform sampling)
    AABBTree           ///< the order is determined so to put close in space points in close indices (optimal for compression)
};

/// squared value
template <typename T>
constexpr inline T sqr( T x ) noexcept { return x * x; }

/// sign of given value in { -1, 0, 1 }
template <typename T>
constexpr inline int sgn( T x ) noexcept { return x > 0 ? 1 : ( x < 0 ? -1 : 0 ); }

/// absolute difference between two value
template <typename T>
constexpr inline T distance( T x, T y ) noexcept { return x >= y ? x - y : y - x; }

/// squared difference between two value
template <typename T>
constexpr inline T distanceSq( T x, T y ) noexcept { return sqr( x - y ); }

/// Linear interpolation: returns v0 when t==0 and v1 when t==1
template <typename V, typename T>
constexpr inline auto lerp( V v0, V v1, T t ) noexcept { return ( 1 - t ) * v0 + t * v1; }

template<typename...>
inline constexpr bool dependent_false = false;

template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };

// explicit deduction guide (not needed as of C++20, but still needed in Clang)
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

class IFastWindingNumber;
class IPointsToMeshProjector;
class IPointsProjector;

namespace MeshBuilder
{

struct BuildSettings;
struct Triangle;
struct VertDuplication;

} //namespace MeshBuilder

} //namespace MR

#ifdef __cpp_lib_unreachable
#   define MR_UNREACHABLE std::unreachable();
#   define MR_UNREACHABLE_NO_RETURN std::unreachable();
#else
#   ifdef __GNUC__
#       define MR_UNREACHABLE __builtin_unreachable();
#       define MR_UNREACHABLE_NO_RETURN __builtin_unreachable();
#   else
#       include <cassert>
#       define MR_UNREACHABLE { assert( false ); return {}; }
#       define MR_UNREACHABLE_NO_RETURN assert( false );
#   endif
#endif
