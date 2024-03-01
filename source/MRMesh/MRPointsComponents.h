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
[[nodiscard]] MRMESH_API Expected<VertBitSet> getLargestComponentsUnion( const PointCloud& pointCloud, float maxDist,
    int minSize, ProgressCallback pc = {} );

/// returns vector of point cloud components containing at least minSize points and connected by a distance no greater than \param maxDist
/// \param minSize must be more than 1
[[nodiscard]] MRMESH_API Expected<std::vector<VertBitSet>> getLargestComponents( const PointCloud& pointCloud, float maxDist,
    int minSize, ProgressCallback pc = {} );

/// gets all components of point cloud connected by a distance no greater than \param maxDist
/// \detail if components are larger than the maxComponentCount, they will be combined into groups of the same size 
/// \note be careful, if point cloud is large enough and has many components, the memory overflow will occur
/// \param maxComponentCount 0 or more then 1. 0 - maximum component count not limited
/// \return pair <components bitsets vector, number components in one group>
[[nodiscard]] MRMESH_API Expected<std::pair<std::vector<VertBitSet>, int>> getAllComponents( const PointCloud& pointCloud, float maxDist, int maxComponentCount = 0,
    ProgressCallback pc = {} );

/// gets union - find structure for vertices in \param region connected by a distance no greater than \param maxDist 
[[nodiscard]] MRMESH_API Expected<UnionFind<VertId>> getUnionFindStructureVerts( const PointCloud& pointCloud, float maxDist, const VertBitSet* region = nullptr, ProgressCallback pc = {} );

/// \}
} // namespace PointCloudComponents

}
