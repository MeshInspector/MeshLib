#pragma once
#include "MRMeshFwd.h"

namespace MR
{
struct PointCloud;

/// \brief Finds radius of ball that avg points in the radius is close to avgPoints parameter
/// \ingroup PointCloudGroup
MRMESH_API float findAvgPointsRadius( const PointCloud& pointCloud, int avgPoints );
}