#pragma once

#include "MRMeshPart.h"

namespace MR
{

/// finds the vertex in the mesh part having the largest projection on given direction,
/// uses aabb-tree inside for faster computation
/// \ingroup AABBTreeGroup
MRMESH_API VertId findDirMax( const Vector3f & dir, const MeshPart & mp );

} //namespace MR
