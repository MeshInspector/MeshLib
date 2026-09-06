#pragma once
#include "MRMeshFwd.h"
#include "MRMesh.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"

namespace MR
{

/// Creates Delaunay triangulation using only XY components of points 
/// points will be changed inside this function take argument by value
[[nodiscard]] MRMESH_API Expected<Mesh> terrainTriangulation( std::vector<Vector3f> points, const ProgressCallback& cb = {} );

/// Creates Delaunay triangulation of valid points of the cloud using only XY components of points,
/// vertex ids in the resulting mesh are the same as in the cloud (invalid points become invalid vertices)
[[nodiscard]] MRMESH_API Expected<Mesh> terrainTriangulation( const PointCloud& cloud, const ProgressCallback& cb = {} );

/// same as above, but moves cloud points into the resulting mesh instead of copying them
[[nodiscard]] MRMESH_API Expected<Mesh> terrainTriangulation( PointCloud&& cloud, const ProgressCallback& cb = {} );

}