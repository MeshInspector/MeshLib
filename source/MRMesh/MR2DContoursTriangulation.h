#pragma once
#include "MRMeshFwd.h"
#include "MRId.h"
#include <memory>
#include <optional>

namespace MR
{

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

/// keeps the internal buffers of the sweep-line triangulation alive between runs,
/// so a caller triangulating many contour sets one by one avoids re-allocating them on every call;
/// one cache must not be used by several threads at once
class ISweepLineCache
{
public:
    /// explicitly define ctors to avoid warning C5267: definition of implicit copy constructor is deprecated because it has a user-provided destructor
    ISweepLineCache() = default;
    ISweepLineCache( const ISweepLineCache & ) = default;
    ISweepLineCache( ISweepLineCache && ) noexcept = default;
    /// pure to make the class abstract: instances are created by makeSweepLineCache() only
    MRMESH_API virtual ~ISweepLineCache() = 0;
};

/// creates a cache for the sweep-line triangulation
MRMESH_API std::unique_ptr<ISweepLineCache> makeSweepLineCache();

/// scratch buffers living in the cache for a caller composing a pipeline around the triangulation
/// (e.g. tracking the hole loops to triangulate), so its per-call locals do not allocate either;
/// the loops scratch is never touched by the triangulation itself, the patch map scratch is where
/// triangulateDisjointContours*( ..., outPatchMap = nullptr, cache ) leaves the patch->input map
/// of the last run. C++-only: the references point inside the cache and must not outlive it.
MR_BIND_IGNORE MRMESH_API EdgeLoops& sweepCacheLoops( ISweepLineCache& cache );
MR_BIND_IGNORE MRMESH_API WholeEdgeMap& sweepCachePatchMap( ISweepLineCache& cache );

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

/// same as triangulateDisjointContours( mesh, loops, normal, outPatchMap ) above, but returns only the patch
/// connectivity, which lives inside \p cache until the next run on it (nullptr if the loops self-intersect);
/// intended for planning: the patch vertex coordinates are not returned
// This is skipped in the bindings: the result points inside the cache and must not outlive it.
MR_BIND_IGNORE MRMESH_API MeshTopology* triangulateDisjointContoursTopology( const Mesh& mesh, const EdgeLoops& loops, const Vector3f& normal, WholeEdgeMap* outPatchMap, ISweepLineCache& cache );

}
}