#pragma once
#include "MRMeshFwd.h"
#include "MRPointCloud.h"

namespace MR
{
MRMESH_API PointCloud meshToPointCloud( const Mesh& mesh, bool saveNormals = true, const VertBitSet* verts = nullptr);
}
