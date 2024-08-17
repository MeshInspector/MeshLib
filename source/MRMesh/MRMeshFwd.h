#pragma once

#include "exports.h"

#include "config.h"

#include "MRKernelFwd.h"
#include "MRBaseFwd.h"

namespace MR
{

class MRMESH_CLASS VoxelTag;
class MRMESH_CLASS RegionTag;
class MRMESH_CLASS NodeTag;
class MRMESH_CLASS ObjTag;
class MRMESH_CLASS TextureTag;

using VoxelId = Id<VoxelTag>;
using RegionId = Id<RegionTag>;
using NodeId = Id<NodeTag>;
using ObjId = Id<ObjTag>;
using TextureId = Id<TextureTag>;

class ViewportId;
class ViewportMask;

/// three vertex ids describing a triangle topology
using ThreeVertIds = std::array<VertId, 3>;
/// mapping from FaceId to a triple of vertex indices
using Triangulation = Vector<ThreeVertIds, FaceId>;

struct UnorientedTriangle;
struct SomeLocalTriangulations;
struct AllLocalTriangulations;

using EdgePath = std::vector<EdgeId>;
using EdgeLoop = std::vector<EdgeId>;

struct MRMESH_CLASS Dipole;
using Dipoles = Vector<Dipole, NodeId>;

using VoxelBitSet = TaggedBitSet<VoxelTag>;
using RegionBitSet = TaggedBitSet<RegionTag>;
using NodeBitSet = TaggedBitSet<NodeTag>;
using ObjBitSet = TaggedBitSet<ObjTag>;
using TextureBitSet = TaggedBitSet<TextureTag>;

template <typename T> class BestFitParabola;
using BestFitParabolaf = BestFitParabola<float>;
using BestFitParabolad = BestFitParabola<double>;

class PointAccumulator;

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

struct PointOnFace;
struct PointOnObject;
struct MeshTriPoint;
struct MeshProjectionResult;
struct MeshIntersectionResult;

/// Coordinates on texture
/// \param x,y should be in range [0..1], otherwise result depends on wrap type of texture (no need to clamp it, it is done on GPU if wrap type is "Clamp" )
using UVCoord = Vector2f;

using ObjMap = Vector<ObjId, ObjId>;
///  mapping of whole edges: map[e]->f, map[e.sym()]->f.sym(), where only map[e] for even edges is stored
using WholeEdgeMap = Vector<EdgeId, UndirectedEdgeId>;
using UndirectedEdge2RegionMap = Vector<RegionId, UndirectedEdgeId>;
using Face2RegionMap = Vector<RegionId, FaceId>;
using Vert2RegionMap = Vector<RegionId, VertId>;

using VertCoords = Vector<Vector3f, VertId>;
using VertNormals = Vector<Vector3f, VertId>;
using VertUVCoords = Vector<UVCoord, VertId>;
using FaceNormals = Vector<Vector3f, FaceId>;

using VertPredicate = std::function<bool( VertId )>;
using FacePredicate = std::function<bool( FaceId )>;
using EdgePredicate = std::function<bool( EdgeId )>;
using UndirectedEdgePredicate = std::function<bool( UndirectedEdgeId )>;

using PreCollapseCallback = std::function<bool( EdgeId edgeToCollapse, const Vector3f& newEdgeOrgPos )>;

using VertMetric = std::function<float( VertId )>;
using FaceMetric = std::function<float( FaceId )>;
using EdgeMetric = std::function<float( EdgeId )>;
using UndirectedEdgeMetric = std::function<float( UndirectedEdgeId )>;

using FaceHashSet = HashSet<FaceId>;
using VertHashSet = HashSet<VertId>;
using EdgeHashSet = HashSet<EdgeId>;

using FaceHashMap = HashMap<FaceId, FaceId>;
using VertHashMap = HashMap<VertId, VertId>;
using EdgeHashMap = HashMap<EdgeId, EdgeId>;
using UndirectedEdgeHashMap = HashMap<UndirectedEdgeId, UndirectedEdgeId>;
///  mapping of whole edges: map[e]->f, map[e.sym()]->f.sym(), where only map[e] for even edges is stored
using WholeEdgeHashMap = HashMap<UndirectedEdgeId, EdgeId>;

class MRMESH_CLASS MeshTopology;
struct MRMESH_CLASS Mesh;
struct MRMESH_CLASS MeshPart;
class MRMESH_CLASS MeshOrPoints;
struct MRMESH_CLASS PointCloud;
class MRMESH_CLASS AABBTree;
class MRMESH_CLASS AABBTreePoints;
class MRMESH_CLASS AABBTreeObjects;
struct MRMESH_CLASS CloudPartMapping;
struct MRMESH_CLASS PartMapping;
struct MeshOrPointsXf;
struct MeshTexture;
struct GridSettings;
struct TriMesh;

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

using GcodeSource = std::vector<std::string>;

class Object;
class SceneRootObject;
class VisualObject;
class ObjectMeshHolder;
class ObjectMesh;
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

struct Image;
class AnyVisualizeMaskEnum;

template <typename T>
struct VoxelsVolume;

template <typename T>
struct VoxelsVolumeMinMax;

using SimpleVolume = VoxelsVolumeMinMax<std::vector<float>>;
using SimpleVolumeU16 = VoxelsVolumeMinMax<std::vector<uint16_t>>;

template <typename T>
using VoxelValueGetter = std::function<T ( const Vector3i& )>;
using FunctionVolume = VoxelsVolume<VoxelValueGetter<float>>;
using FunctionVolumeU8 = VoxelsVolume<VoxelValueGetter<uint8_t>>;

#ifndef MRMESH_NO_OPENVDB
class ObjectVoxels;

struct OpenVdbFloatGrid;
using FloatGrid = std::shared_ptr<OpenVdbFloatGrid>;
using VdbVolume = VoxelsVolumeMinMax<FloatGrid>;
#endif

class HistoryAction;
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

using VertColorMapAggregator = ColorMapAggregator<VertTag>;
using UndirEdgeColorMapAggregator = ColorMapAggregator<UndirectedEdgeTag>;
using FaceColorMapAggregator = ColorMapAggregator<FaceTag>;

class WatershedGraph;

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

template<typename...>
inline constexpr bool dependent_false = false;

class IFastWindingNumber;
class IPointsToMeshProjector;

namespace MeshBuilder
{

struct BuildSettings;
struct Triangle;
struct VertDuplication;

} //namespace MeshBuilder

} //namespace MR
