#pragma once
#include "MRMeshFwd.h"
#include "MRMesh.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"

namespace MR
{

/// Creates Delaunay triangulation of given points projected on XY plane,
/// the vertices of the resulting mesh have the original 3D coordinates and the same ids as the input points;
/// points will be moved inside this function, so the argument is taken by value
[[nodiscard]] MRMESH_API Expected<Mesh> delaunayTriangulationXY( std::vector<Vector3f> points, const ProgressCallback& cb = {} );

/// Creates Delaunay triangulation of valid points of the cloud projected on XY plane,
/// vertex ids in the resulting mesh are the same as in the cloud (invalid points become invalid vertices)
[[nodiscard]] MRMESH_API Expected<Mesh> delaunayTriangulationXY( const PointCloud& cloud, const ProgressCallback& cb = {} );

/// same as above, but moves cloud points into the resulting mesh instead of copying them
[[nodiscard]] MRMESH_API Expected<Mesh> delaunayTriangulationXY( PointCloud&& cloud, const ProgressCallback& cb = {} );

}
