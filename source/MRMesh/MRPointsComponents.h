#pragma once
#include "MRMeshFwd.h"
#include "MRUnionFind.h"
#include "MRExpected.h"

namespace MR
{

namespace PointCloudComponents
{

/// \defgroup PointClodComponentsGroup PointClodComponents
/// \ingroup ComponentsGroup
/// \{

/// get point cloud components contain more then minSize points and connected by a distance no greater than \param maxDist
/// \param minSize if it less then 0, minSize = 1% of valid points
MRMESH_API Expected<std::vector<VertBitSet>> getLargestComponents( const PointCloud& pointCloud, float maxDist, int minSize = -1, ProgressCallback pc = {} );

/// gets union - find structure for vertices in \param region connected by a distance no greater than \param maxDist 
MRMESH_API Expected<UnionFind<VertId>> getUnionFindStructureVerts( const PointCloud& pointCloud, float maxDist, const VertBitSet* region = nullptr, ProgressCallback pc = {} );

/// \}
} // namespace PointClodComponents

}
