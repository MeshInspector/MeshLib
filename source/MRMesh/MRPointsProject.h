#pragma once
#include "MRMeshFwd.h"
#include "MRId.h"
#include <cfloat>

namespace MR
{
/// \addtogroup AABBTreeGroup
/// \{

struct PointsProjectionResult
{
    /// the closest vertex in point cloud
    VertId vId;
    /// squared distance from pt to proj
    float distSq{ 0 };
};

/**
 * \brief computes the closest point on point cloud to given point
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
 * \param xf pointcloud-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 */
[[nodiscard]] MRMESH_API PointsProjectionResult findProjectionOnPoints( const Vector3f& pt, const PointCloud& pc,
    float upDistLimitSq = FLT_MAX,
    const AffineXf3f* xf = nullptr,
    float loDistLimitSq = 0 );

/// \}
}