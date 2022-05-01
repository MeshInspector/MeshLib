#pragma once
#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRId.h"
#include <list>
#include <climits>

namespace MRE
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
MREALGORITHMS_API float updateNeighborsRadius( const MR::VertCoords& points, MR::VertId v, const std::list<MR::VertId>& fan, float baseRadius );

/**
 * \brief Finds all neighbors of v in given radius (v excluded)
 * \ingroup TriangulationHelpersGroup
 */
MREALGORITHMS_API std::vector<MR::VertId> findNeighbors( const MR::PointCloud& pointCloud, MR::VertId v, float radius );

/**
 * \brief Result of local triangulation
 * \ingroup TriangulationHelpersGroup
 */
struct TriangulatedFan
{
    std::list<MR::VertId> optimized; ///< Good neighbors after optimization (each pair is fan triangle)
    MR::VertId border; ///< First border edge (triangle associated with this point is absent)
};

/** 
 * \brief Creates local triangulation by sorting and optimizing neighbors fan (normal of v is needed for consistent fan orientation)
 * \ingroup TriangulationHelpersGroup
 * 
 * \param critAngle max allowed angle for triangles in fan
 * \param steps max optimization steps (INT_MAX - default)
 */
MREALGORITHMS_API TriangulatedFan trianglulateFan( const MR::VertCoords& points, MR::VertId v, const std::vector<MR::VertId>& neighbors,
    const MR::VertCoords& normals, float critAngle, int steps = INT_MAX );

}

}
