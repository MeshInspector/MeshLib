#pragma once
#include "MRMeshFwd.h"
#include "MRBitSet.h"
#include "MRId.h"

namespace MR
{
struct PointCloud;

using FoundPointCallback = std::function<void( VertId, const Vector3f& )>;

/// Finds all valid points of pointCloud that are inside given ball (distance to center are lower then radius)
/// \ingroup AABBTreeGroup
MRMESH_API void findPointsInBall( const PointCloud& pointCloud, const Vector3f& center, float radius, const FoundPointCallback& foundCallback );
}