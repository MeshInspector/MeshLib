#pragma once
#include "MRMeshFwd.h"
#include "MRUnionFind.h"

namespace MR
{

namespace PointClodComponents
{

/// \defgroup PointClodComponentsGroup PointClodComponents
/// \ingroup ComponentsGroup
/// \{

/// get all point cloud components connected by a distance no greater than \param maxDist
std::vector<VertBitSet> getAllComponents( const PointCloud& pointCloud, float maxDist );

/// gets union - find structure for vertices in \param region connected by a distance no greater than \param maxDist 
UnionFind<VertId> getUnionFindStructureVerts( const PointCloud& pointCloud, float maxDist, const VertBitSet* region = nullptr );

/// \}
} // namespace PointClodComponents

}
