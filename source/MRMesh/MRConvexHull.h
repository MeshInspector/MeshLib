#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// computes the mesh of convex hull from given input points
MRMESH_API Mesh makeConvexHull( const VertCoords & points, const VertBitSet & validPoints );
MRMESH_API Mesh makeConvexHull( const Mesh & in );
MRMESH_API Mesh makeConvexHull( const PointCloud & in );

} //namespace MR
