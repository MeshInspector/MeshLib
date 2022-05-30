#pragma once

#include "MRVector.h"
#include "MRId.h"
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
MRMESH_API float updateNeighborsRadius( const VertCoords& points, VertId v, const std::list<VertId>& fan, float baseRadius );

/**
 * \brief Finds all neighbors of v in given radius (v excluded)
 * \ingroup TriangulationHelpersGroup
 */
MRMESH_API std::vector<VertId> findNeighbors( const PointCloud& pointCloud, VertId v, float radius );

/**
 * \brief Result of local triangulation
 * \ingroup TriangulationHelpersGroup
 */
struct TriangulatedFan
{
    std::list<VertId> optimized; ///< Good neighbors after optimization (each pair is fan triangle)
    VertId border; ///< First border edge (triangle associated with this point is absent)
};

/** 
 * \brief Creates local triangulation by sorting and optimizing neighbors fan (normal of v is needed for consistent fan orientation)
 * \ingroup TriangulationHelpersGroup
 * 
 * \param critAngle max allowed angle for triangles in fan
 * \param steps max optimization steps (INT_MAX - default)
 */
MRMESH_API TriangulatedFan trianglulateFan( const VertCoords& points, VertId v, const std::vector<VertId>& neighbors,
    const VertCoords& normals, float critAngle, int steps = INT_MAX );

}

} //namespace MR
