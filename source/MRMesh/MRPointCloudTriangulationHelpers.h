#pragma once

#include "MRVector.h"
#include "MRId.h"
#include "MRConstants.h"
#include "MRBuffer.h"
#include <climits>
#include <optional>

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
MRMESH_API void findNeighbors( const PointCloud& pointCloud, VertId v, float radius, std::vector<VertId>& neighbors );

/**
 * \brief Filter neighbors with crossing normals
 * \ingroup TriangulationHelpersGroup
 */
MRMESH_API void filterNeighbors( const VertNormals& normals, VertId v, std::vector<VertId>& neighbors );

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
};

/** 
 * \brief Creates local triangulation by sorting and optimizing neighbors fan (normal of v is needed for consistent fan orientation)
 * \ingroup TriangulationHelpersGroup
 * 
 * \param critAngle max allowed angle for triangles in fan
 * \param trustedNormals if not null, contains valid oriented normals of all points
 * \param steps max optimization steps (INT_MAX - default)
 */
MRMESH_API void trianglulateFan( const VertCoords& points, VertId v, TriangulatedFanData& triangulationData,
    const VertCoords* trustedNormals, float critAngle, int steps = INT_MAX );

struct Settings
{
    /// initial radius of search for neighbours, it can be increased automatically
    float radius = 0;
    /// max allowed angle for triangles in fan
    float critAngle = PI2_F;
    /// if oriented normals are known, they will be used for neighbour points selection
    const VertCoords* trustedNormals = nullptr;
    /// automatic increase of the radius if points outside can make triangles from original radius not-Delone
    bool automaticRadiusIncrease = true;
    /// the maximum number of optimization steps (removals) in local triangulation
    int maxRemoves = INT_MAX;
    /// optional output of considered neighbor points after filtering but before triangulation/optimization
    std::vector<VertId> * allNeighbors = nullptr;
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
 * \brief Checks if given vertex is on boundary of the point cloud
 * \details The vertex is considered as boundary if its neighbor ring has angle more than \param boundaryAngle degrees
 * \ingroup TriangulationHelpersGroup
 * \param pointCloud input point cloud
 * \param normals non-oriented normals for the point cloud
 * \param v vertex id to check
 * \param radius radius to find neighbors in
 * \param triangulationData cache structure for neighbors, not to allocate for multiple calls
 * \returns true if vertex is boundary, false otherwise
 */
MRMESH_API bool isBoundaryPoint( const PointCloud& pointCloud, const VertCoords& normals, 
    VertId v, float radius, float boundaryAngle,
    TriangulatedFanData& triangulationData );
}

} //namespace MR
