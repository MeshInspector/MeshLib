#pragma once
#include "MRMeshFwd.h"
#include "MRBitSet.h"
#include "MRId.h"

namespace MR
{

using FoundPointCallback = std::function<void( VertId, const Vector3f& )>;

/// Finds all valid points of pointCloud that are inside or on the surface of given ball (center, radius)
/// \ingroup AABBTreeGroup
MRMESH_API void findPointsInBall( const PointCloud& pointCloud, const Vector3f& center, float radius, const FoundPointCallback& foundCallback );

/// Finds all points in tree that are inside or on the surface of given ball (center, radius)
/// \ingroup AABBTreeGroup
MRMESH_API void findPointsInBall( const AABBTreePoints& tree, const Vector3f& center, float radius, const FoundPointCallback& foundCallback );

}