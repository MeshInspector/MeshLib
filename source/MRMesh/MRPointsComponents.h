#pragma once
#include "MRMeshFwd.h"
#include "MRUnionFind.h"
#include "MRExpected.h"

namespace MR
{

namespace PointCloudComponents
{

/// \defgroup PointCloudComponentsGroup PointCloudComponents
/// \ingroup ComponentsGroup
/// \{

/// returns the union of point cloud components containing at least minSize points and connected by a distance no greater than \param maxDist
/// \param minSize must be more than 1
MRMESH_API Expected<VertBitSet> getLargestComponentsUnion( const PointCloud& pointCloud, float maxDist, int minSize, ProgressCallback pc = {} );

/// returns vector of point cloud components containing at least minSize points and connected by a distance no greater than \param maxDist
/// \param minSize must be more than 1
MRMESH_API Expected<std::vector<VertBitSet>> getLargestComponents( const PointCloud& pointCloud, float maxDist, int minSize, ProgressCallback pc = {} );

/// gets union - find structure for vertices in \param region connected by a distance no greater than \param maxDist 
MRMESH_API Expected<UnionFind<VertId>> getUnionFindStructureVerts( const PointCloud& pointCloud, float maxDist, const VertBitSet* region = nullptr, ProgressCallback pc = {} );

/// \}
} // namespace PointCloudComponents

}
