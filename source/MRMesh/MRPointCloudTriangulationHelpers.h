#pragma once

#include "MRPch/MRBindingMacros.h"
#include "MRVector.h"
#include "MRId.h"
#include "MRConstants.h"
#include "MRBuffer.h"
#include "MRPointsProject.h"
#include "MRFewSmallest.h"
#include <climits>
#include <optional>
#include <queue>

namespace MR
{
/**
 * \brief Collection of functions and structures needed for PointCloud triangulation
 * \defgroup TriangulationHelpersGroup TriangulationHelpers
 * \ingroup PointCloudTriangulationGroup
 */
namespace TriangulationHelpers
{

/**
 * \brief Finds max radius of neighbors search, for possible better local triangulation
 * \param borderV first boundary vertex in \param fan (next VertId in fan is also boundary but first is enough)
 * \ingroup TriangulationHelpersGroup
 */
MRMESH_API float updateNeighborsRadius( const VertCoords& points, VertId v, VertId boundaryV,
    const std::vector<VertId>& fan, float baseRadius );

/**
 * \brief Finds all neighbors of v in given radius (v excluded)
 * \ingroup TriangulationHelpersGroup
 */
MRMESH_API void findNeighborsInBall( const PointCloud& pointCloud, VertId v, float radius, std::vector<VertId>& neighbors );

/**
 * \brief Finds at most given number of neighbors of v (v excluded)
 * \param tmp temporary storage to avoid its allocation
 * \param upDistLimitSq upper limit on the distance in question, points with larger distance than it will not be returned
 * \return maxDistSq to the furthest returned neighbor (or 0 if no neighbours are returned)
 * \ingroup TriangulationHelpersGroup
 */
MRMESH_API float findNumNeighbors( const PointCloud& pointCloud, VertId v, int numNeis, std::vector<VertId>& neighbors,
    FewSmallest<PointsProjectionResult> & tmp, float upDistLimitSq = FLT_MAX );

/**
 * \brief Filter neighbors with crossing normals
 * \ingroup TriangulationHelpersGroup
 */
MRMESH_API void filterNeighbors( const VertNormals& normals, VertId v, std::vector<VertId>& neighbors );

struct FanOptimizerQueueElement
{
    float weight{ 0.0f }; // profit of flipping this edge
    int id{ -1 }; // index

    // needed to remove outdated queue elements
    int prevId{ -1 }; // id of prev neighbor
    int nextId{ -1 }; // id of next neighbor

    bool stable{ false }; // if this flag is true, edge cannot be flipped
    bool operator < ( const FanOptimizerQueueElement& other ) const
    {
        if ( stable == other.stable )
            return weight < other.weight;
        return stable;
    }
    bool operator==( const FanOptimizerQueueElement& other ) const = default;

    bool isOutdated( const std::vector<VertId>& neighbors ) const
    {
        return !neighbors[nextId].valid() || !neighbors[prevId].valid();
    }
};

/**
 * \brief Data with caches for optimizing fan triangulation
 * \ingroup TriangulationHelpersGroup
 */
struct TriangulatedFanData
{
    /// clockwise points around center point in (optimized) triangle fan,
    /// each pair of points (as well as back()-front() pair) together with the center form a fan triangle
    std::vector<VertId> neighbors;

    /// temporary reusable storage to avoid allocations for each point
    std::vector<std::pair<double, int>> cacheAngleOrder;

    /// first border edge (invalid if the center point is not on the boundary)
    /// triangle associated with this point is absent
    VertId border;

    /// the storage to collect n-nearest neighbours, here to avoid allocations for each point
    FewSmallest<PointsProjectionResult> nearesetPoints;

    /// the queue to optimize local triangulation, here to avoid allocations for each point
    /// Removed from the bindings because we don't have bindings for `std::priority_queue` at the moment.
    MR_BIND_IGNORE std::priority_queue<FanOptimizerQueueElement> queue;
};

struct Settings
{
    /// initial radius of search for neighbours, it can be increased automatically;
    /// if radius is positive then numNeis must be zero
    float radius = 0;

    /// initially selects given number of nearest neighbours;
    /// if numNeis is positive then radius must be zero
    int numNeis = 0;

    /// max allowed angle for triangles in fan
    float critAngle = PI2_F;

    /// the vertex is considered as boundary if its neighbor ring has angle more than this value
    float boundaryAngle = 0.9f * PI_F;

    /// if oriented normals are known, they will be used for neighbor points selection
    const VertCoords* trustedNormals = nullptr;

    /// automatic increase of the radius if points outside can make triangles from original radius not-Delone
    bool automaticRadiusIncrease = true;

    /// the maximum number of optimization steps (removals) in local triangulation
    int maxRemoves = INT_MAX;

    /// optional output of considered neighbor points after filtering but before triangulation/optimization
    std::vector<VertId> * allNeighbors = nullptr;

    /// optional output: actual radius of neighbor search (after increase if any)
    float * actualRadius = nullptr;

    /// optional: if provided this cloud will be used for searching of neighbors (so it must have same validPoints)
    const PointCloud * searchNeighbors = nullptr;
};

/// constructs local triangulation around given point
MRMESH_API void buildLocalTriangulation( const PointCloud& cloud, VertId v, const Settings & settings,
    TriangulatedFanData & fanData );

/// computes all local triangulations of all points in the cloud, and returns them distributed among
/// a set of SomeLocalTriangulations objects
[[nodiscard]] MRMESH_API std::optional<std::vector<SomeLocalTriangulations>> buildLocalTriangulations(
    const PointCloud& cloud, const Settings & settings, const ProgressCallback & progress = {} );

//// computes local triangulations of all points in the cloud united in one struct
[[nodiscard]] MRMESH_API std::optional<AllLocalTriangulations> buildUnitedLocalTriangulations(
    const PointCloud& cloud, const Settings & settings, const ProgressCallback & progress = {} );

/**
 * \brief Checks if given vertex is on boundary of the point cloud, by constructing local triangulation around it
 * \ingroup TriangulationHelpersGroup
 * \param cloud input point cloud
 * \param v vertex id to check
 * \param settings all parameters of the computation
 * \param fanData cache structure for neighbors, not to allocate for multiple calls
 * \returns true if vertex is boundary, false otherwise
 */
[[nodiscard]] MRMESH_API bool isBoundaryPoint( const PointCloud& cloud, VertId v, const Settings & settings,
    TriangulatedFanData & fanData );

/// Returns bit set of points that are considered as boundary by calling isBoundaryPoint in each
[[nodiscard]] MRMESH_API std::optional<VertBitSet> findBoundaryPoints( const PointCloud& pointCloud, const Settings & settings,
    ProgressCallback cb = {} );


} //namespace TriangulationHelpers

} //namespace MR
