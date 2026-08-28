#pragma once
#include "MRMeshFwd.h"
#include "MRPointCloud.h"
#include "MRExpected.h"
#include "MRProgressCallback.h"

namespace MR
{

///  Mesh to PointCloud
/// \ingroup MeshAlgorithmGroup
MRMESH_API PointCloud meshToPointCloud( const Mesh& mesh, bool saveNormals = true, const VertBitSet* verts = nullptr);

/// Converts the mesh or its part in a point cloud dense enough to stop any ball of given radius:
/// no ball of the radius can pass through the sampled surface without touching at least one point of
/// the cloud, because every point of that surface is within the radius from some point of the cloud.
/// The cloud consists of
/// 1) all vertices of the sampled faces, having the same ids as in the mesh;
/// 2) samples on each edge longer than 2*radius;
/// 3) samples inside each triangle, which cannot be covered by its vertices alone.
/// Please note that the number of samples grows as 1/radius^2.
/// \param mp the mesh or the part of it to be covered; nothing outside the part is sampled
/// \param saveNormals if true then the normals of the cloud are set as well: the normals of the mesh
///        vertices, and their interpolation in the samples on the edges and inside the triangles
/// \ingroup MeshAlgorithmGroup
[[nodiscard]] MRMESH_API Expected<PointCloud> meshToDensePointCloud( const MeshPart& mp, float radius,
    bool saveNormals = true, const ProgressCallback& cb = {} );

}
