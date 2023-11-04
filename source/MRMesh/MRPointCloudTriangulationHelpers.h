#pragma once

#include "MRVector.h"
#include "MRId.h"
#include "MRConstants.h"
#include <list>
#include <climits>

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
 * \ingroup TriangulationHelpersGroup
 */
MRMESH_API float updateNeighborsRadius( const VertCoords& points, VertId v, const std::vector<VertId>& fan, float baseRadius );

/**
 * \brief Finds all neighbors of v in given radius (v excluded)
 * \ingroup TriangulationHelpersGroup
 */
MRMESH_API void findNeighbors( const PointCloud& pointCloud, VertId v, float radius, std::vector<VertId>& neighbors );

/**
 * \brief Data with caches for optimizing fan triangulation
 * \ingroup TriangulationHelpersGroup
 */
struct TriangulatedFanData
{
    std::vector<VertId> neighbors; ///< algorithm reorders this vector to be optimized (each pair is fan triangle)
    std::vector<std::pair<double, int>> cacheAngleOrder; ///< cache vector for ordering neighbors
    VertId border; ///< First border edge (triangle associated with this point is absent)
};

/** 
 * \brief Creates local triangulation by sorting and optimizing neighbors fan (normal of v is needed for consistent fan orientation)
 * \ingroup TriangulationHelpersGroup
 * 
 * \param critAngle max allowed angle for triangles in fan
 * \param steps max optimization steps (INT_MAX - default)
 */
MRMESH_API void trianglulateFan( const VertCoords& points, VertId v, TriangulatedFanData& triangulationData,
    const VertCoords& normals, float critAngle, int steps = INT_MAX );

struct Settings
{
    /// initial radius of search for neighbours, it can be increased automatically
    float radius = 0;
    /// max allowed angle for triangles in fan
    float critAngle = PI2_F;
};

/// constructs local triangulation around given point with automatic increase of the radius
MRMESH_API void buildLocalTriangulation( const PointCloud& cloud, VertId v, const VertCoords& normals, const Settings & settings,
    TriangulatedFanData & fanData );

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
