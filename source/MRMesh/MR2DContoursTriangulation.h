#pragma once
#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRHoleFillPlan.h"
#include <memory>
#include <optional>

namespace MR
{

struct PreciseVertCoords;
struct PreciseCutPoint;
class CutTrianglePredicates;

namespace PlanarTriangulation
{

/// Specify mode of detecting inside and outside parts of triangulation
enum class WindingMode
{
    NonZero,
    Positive,
    Negative
};

/// Info about intersection point for mapping
struct IntersectionInfo
{
    /// if lDest is invalid then lOrg is id of input vertex
    /// ids of lower intersection edge vertices
    VertId lOrg, lDest;
    /// ids of upper intersection edge vertices
    VertId uOrg, uDest;

    // ratio of intersection
    // 0.0 -> point is lOrg
    // 1.0 -> point is lDest
    float lRatio = 0.0f;
    // 0.0 -> point is uOrg
    // 1.0 -> point is uDest
    float uRatio = 0.0f;
    bool isIntersection() const { return lDest.valid(); }
};

using ContourIdMap = std::vector<IntersectionInfo>;
using ContoursIdMap = std::vector<ContourIdMap>;

/// struct to map new vertices (only appear on intersections) of the outline to it's edges
struct IntersectionsMap
{
    /// shift of index
    size_t shift{ 0 };
    /// map[id-shift] = {lower intersection edge, upper intersection edge}
    ContourIdMap map;
};

struct BaseOutlineParameters
{
    bool allowMerge{ false }; ///< allow to merge vertices with same coordinates
    WindingMode innerType{ WindingMode::Negative }; ///< what to mark as inner part
};

/// returns Mesh with boundaries representing outline if input contours
/// interMap optional output intersection map
MRMESH_API Mesh getOutlineMesh( const Contours2f& contours, IntersectionsMap* interMap = nullptr, const BaseOutlineParameters& params = {} );
MRMESH_API Mesh getOutlineMesh( const Contours2d& contours, IntersectionsMap* interMap = nullptr, const BaseOutlineParameters& params = {} );

struct OutlineParameters
{
    ContoursIdMap* indicesMap{ nullptr }; ///< optional output from result contour ids to input ones
    BaseOutlineParameters baseParams;
};

/// returns Contour representing outline if input contours
MRMESH_API Contours2f getOutline( const Contours2f& contours, const OutlineParameters& params = {} );
MRMESH_API Contours2f getOutline( const Contours2d& contours, const OutlineParameters& params = {} );

struct TriangulationParameters
{
    /// optional output: winding number of the region each face belongs to;
    /// when set, Delone flips after triangulation are skipped, so each face stays strictly inside one winding region
    Vector<int, FaceId>* outFaceWinding{ nullptr };

    /// optional output: maps each vertex created at contours intersection to the pair of intersected edges;
    /// vertices with id less than `shift` are original contour vertices
    IntersectionsMap* outInterMap{ nullptr };
};

/**
 * @brief triangulate 2d contours
 * only closed contours are allowed (first point of each contour should be the same as last point of the contour)
 * @return return created mesh
 */
MRMESH_API Mesh triangulateContours( const Contours2d& contours, const TriangulationParameters& params = {} );
MRMESH_API Mesh triangulateContours( const Contours2f& contours, const TriangulationParameters& params = {} );

/// keeps the internal buffers of the sweep-line triangulation alive between runs, so a caller
/// triangulating many contour sets one by one avoids re-allocating them on every call;
/// the entries returning a Mesh move the topology and the points out into it, so those two grow again
/// on the next run, while \ref triangulateDisjointContoursTopology keeps everything;
/// one cache must not be used by several threads at once
class ISweepLineCache
{
public:
    ISweepLineCache() = default;
    /// not copyable: the cache is its owner's private scratch, and a copy would silently duplicate
    /// every buffer (this also silences warning C5267 about the user-provided destructor below)
    ISweepLineCache( const ISweepLineCache & ) = delete;
    ISweepLineCache & operator =( const ISweepLineCache & ) = delete;
    /// pure to make the class abstract: instances are created by makeSweepLineCache() only
    MRMESH_API virtual ~ISweepLineCache() = 0;
};

/// creates a cache for the sweep-line triangulation
MRMESH_API std::unique_ptr<ISweepLineCache> makeSweepLineCache();

/**
 * @brief triangulate 2d contours
 * only closed contours are allowed (first point of each contour should be the same as last point of the contour)
 * @return std::optional<Mesh> : if some contours intersect return false, otherwise return created mesh
 */
MRMESH_API std::optional<Mesh> triangulateDisjointContours( const Contours2d& contours, ISweepLineCache* cache = nullptr );
MRMESH_API std::optional<Mesh> triangulateDisjointContours( const Contours2f& contours, ISweepLineCache* cache = nullptr );

/**
 * @brief triangulate hole boundary loops of \p mesh in the mesh's own 3d space, orienting faces around \p normal
 * combinatorics run on the dominant-axis projection of \p normal; output vertices keep the exact mesh coordinates
 * (no projection round-trip), and loops sharing a mesh vertex are merged by identity.
 * @param loops one closed EdgeLoop per contour (as produced by trackRightBoundaryLoop on each hole edge)
 * @param outPatchMap optional output: for each patch boundary edge (by undirected id) the mesh edge it copies,
 *        directed along it; edges past its size are the triangulation's own
 * @return std::nullopt if the loops self-intersect, otherwise the patch mesh
 */
MRMESH_API std::optional<Mesh> triangulateDisjointContours( const Mesh& mesh, const EdgeLoops& loops, const Vector3f& normal, WholeEdgeMap* outPatchMap = nullptr, ISweepLineCache* cache = nullptr );

/// same as triangulateDisjointContours( mesh, loops, normal, &outPatchMap ) above, but returns only the
/// patch connectivity, which lives inside \p cache until the next run on it, and no vertex coordinates;
/// intended for planning
/// \return nullptr if the loops self-intersect; an empty topology if \p loops is empty
// This is skipped in the bindings: the result points inside the cache and must not outlive it.
MR_BIND_IGNORE MRMESH_API MeshTopology* triangulateDisjointContoursTopology( const Mesh& mesh, const EdgeLoops& loops, const Vector3f& normal, WholeEdgeMap& outPatchMap, ISweepLineCache& cache );

/**
 * @brief splits the near-planar region bounded by \p loops in monotone parts, without triangulating them
 * this is the sweep-line triangulation stopped right after monotonation
 * @param loops one closed EdgeLoop per boundary of one region, bounding holes of \p mesh as trackRightBoundaryLoop
 *        returns them (a slit of one edge is a loop of 2 edges, and a loop of one edge is skipped); which way round a loop runs
 *        does not matter; the holes can be several parts of one region split by edges between them, as all the holes of a face
 *        cut along contours (the parts between the contours that cross its sides, and both sides of every island), since an edge
 *        taken both ways adds zero winding
 * @param normal the orientation the region is monotone around, see \ref triangulateDisjointContours
 * @return std::nullopt if the loops intersect or a chord would land in a face instead of a hole, so that
 *         the caller can fall back on filling every loop separately; otherwise a plan with numTris == 0
 *         for \ref executeHoleFillPlan, which then only adds edges (empty if already monotone, a no-op).
 *         Validate it with \ref isFillingMultipleEdgeFree: a chord may duplicate an off-loop mesh edge
 */
MRMESH_API std::optional<HoleFillPlan> getMonotonePlan( const Mesh& mesh, const EdgeLoops& loops, const Vector3f& normal );

/// same as above for the region of one triangle of a mesh cut by the boolean (the loops of all the holes of the triangle),
/// on the exact predicates of \ref CutTrianglePredicates, which never contradict the boolean's ones, so the loops are disjoint
/// whenever the boolean's contours are
/// \param tri the triangle, counterclockwise around its normal
/// \param pointOf the definition of every vertex on the loops, valid as \ref CutTrianglePredicates::addPoint requires
/// \return std::nullopt also if two vertices on the loops are defined alike and their wedges do not tell them apart, or the predicates fail
///         on the points
MR_BIND_IGNORE MRMESH_API std::optional<HoleFillPlan> getMonotonePlan( const Mesh& mesh, const EdgeLoops& loops,
    const std::array<PreciseVertCoords, 3>& tri, const std::function<PreciseCutPoint( VertId )>& pointOf );

/// finds where the edges of the loops of one triangle of a mesh cut by the boolean cross, on the exact predicates of \ref getMonotonePlan above
/// \param cutOf the triangle of the other mesh whose plane every edge of the loops inside the triangle goes along (as PreciseCutPoint::tri),
///        or invalid ids for an edge along a side; every crossing of two cuts is a point of the predicates (see PreciseCutPoint::tri2)
/// \return the crossings in the order a sweep over the triangle meets them, which is also their order along every edge (crossings at one
///         point are met one by one); every crossing is two edges directed along the sweep, the first one passing from the right of the
///         second one to its left; std::nullopt if an edge without a cut crosses another one, if two vertices on the loops are defined alike
///         and their wedges do not tell them apart, if two edges of the loops join the same two vertices, or if the predicates fail on the
///         points or on a crossing
MR_BIND_IGNORE MRMESH_API std::optional<Vector<EdgePair>> findCrossings( const Mesh& mesh, const EdgeLoops& loops,
    const std::array<PreciseVertCoords, 3>& tri, const std::function<PreciseCutPoint( VertId )>& pointOf,
    const std::function<std::array<PreciseVertCoords, 3>( UndirectedEdgeId )>& cutOf );

/// same as above on the predicates of \ref getMonotonePlan( mesh, loops, normal ), as where no exact definitions are given: on the points
/// rounded as there, so the rounding can add or remove crossings, as of an edge with a side of the triangle that it nearly touches
/// \return std::nullopt if two edges of the loops join the same two vertices
MR_BIND_IGNORE MRMESH_API std::optional<Vector<EdgePair>> findCrossings( const Mesh& mesh, const EdgeLoops& loops, const Vector3f& normal );

/// \ref hasCrossing tests up to this many segments in pairs, which is faster for them than the sweep, the only user of its cache with the exact predicates
constexpr size_t cMaxSegmentsInPairs = 32;

/// whether two of the segments between the points of \p preds cross, found by the sweep of \ref triangulateDisjointContours,
/// which stops at the first crossing, or by testing every two of up to \ref cMaxSegmentsInPairs segments;
/// segments with a common end only touch, and a repeated segment is taken once
/// \param segments the ends of every segment, as the ids of the points in \p preds, which are all set and not failed
/// \return true also if the predicates fail
MR_BIND_IGNORE MRMESH_API bool hasCrossing( const Vector<TwoVertIds>& segments, const CutTrianglePredicates& preds, ISweepLineCache* cache = nullptr );

/// same as above for the points that \p segments index given by their coordinates, projected along the axis most aligned with \p normal
/// and rounded to integers as \ref triangulateDisjointContours does; distinct points rounded to the same position are told apart by their ids, not merged;
/// if all the points are at one position, no segments cross; \p cache also keeps the projected points between calls
MR_BIND_IGNORE MRMESH_API bool hasCrossing( const Vector<TwoVertIds>& segments, const VertCoords& points, const Vector3f& normal,
    ISweepLineCache* cache = nullptr );

}
}