#pragma once
#include "MRMeshFwd.h"
#include "MRUnionFind.h"

namespace MR
{

namespace PointCloudComponents
{

/// \defgroup PointClodComponentsGroup PointClodComponents
/// \ingroup ComponentsGroup
/// \{

/// get point cloud components contain more then minSize points and connected by a distance no greater than \param maxDist
/// \param minSize if it less then 0, minSize = 1% of valid points
MRMESH_API std::vector<VertBitSet> getBigComponents( const PointCloud& pointCloud, float maxDist, int minSize = -1 );

/// gets union - find structure for vertices in \param region connected by a distance no greater than \param maxDist 
MRMESH_API UnionFind<VertId> getUnionFindStructureVerts( const PointCloud& pointCloud, float maxDist, const VertBitSet* region = nullptr );

/// \}
} // namespace PointClodComponents

}
