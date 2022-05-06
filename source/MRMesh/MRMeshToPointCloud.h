#pragma once
#include "MRMeshFwd.h"
#include "MRPointCloud.h"

namespace MR
{

///  Mesh to PointCloud
/// \ingroup MeshAlgorithmGroup
MRMESH_API PointCloud meshToPointCloud( const Mesh& mesh, bool saveNormals = true, const VertBitSet* verts = nullptr);
}
