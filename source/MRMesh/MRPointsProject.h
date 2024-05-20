#pragma once
#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRProgressCallback.h"
#include <cfloat>

namespace MR
{
/// \addtogroup AABBTreeGroup
/// \{

struct PointsProjectionResult
{
    /// squared distance from pt to proj
    float distSq{ 0 };
    /// the closest vertex in point cloud
    VertId vId;

    auto operator <=>(const PointsProjectionResult &) const = default;
};


/**
 * \brief computes the closest point on point cloud to given point
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
 * \param xf pointcloud-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 * \param skipCb callback to discard VertId projection candidate
 */
[[nodiscard]] MRMESH_API PointsProjectionResult findProjectionOnPoints( const Vector3f& pt, const PointCloud& pc,
    float upDistLimitSq = FLT_MAX,
    const AffineXf3f* xf = nullptr,
    float loDistLimitSq = 0,
    VertPredicate skipCb = {} );

/**
 * \brief finds a number of the closest points in the cloud (as configured in \param res) to given point
 * \param upDistLimitSq upper limit on the distance in question, points with larger distance than it will not be returned
 * \param xf pointcloud-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, the algorithm can return given number of points within this distance even skipping closer ones
 */
MRMESH_API void findFewClosestPoints( const Vector3f& pt, const PointCloud& pc, FewSmallest<PointsProjectionResult> & res,
    float upDistLimitSq = FLT_MAX,
    const AffineXf3f* xf = nullptr,
    float loDistLimitSq = 0 );

/**
 * \brief finds given number of closest points (excluding itself) to each valid point in the cloud;
 * \param numNei the number of closest points to find for each point
 * \return a buffer where for every valid point with index `i` its neighbours are stored at indices [i*numNei; (i+1)*numNei)
 */
[[nodiscard]] MRMESH_API Buffer<VertId> findNClosestPointsPerPoint( const PointCloud& pc, int numNei, const ProgressCallback & progress = {} );

/// finds two closest points (first id < second id) in whole point cloud
[[nodiscard]] MRMESH_API VertPair findTwoClosestPoints( const PointCloud& pc, const ProgressCallback & progress = {} );

/// \}
}