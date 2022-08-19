#pragma once

#include "MRMeshPart.h"

namespace MR
{

enum class UseAABBTree : char
{
    No,  // AABB-tree of the mesh will not be used, even if it is available
    Yes, // AABB-tree of the mesh will be used even if it has to be constructed
    YesIfAlreadyConstructed, // AABB-tree of the mesh will be used if it was previously constructed and available, and will not be used otherwise
};

/// finds the vertex in the mesh part having the largest projection on given direction,
/// uses aabb-tree inside for faster computation
/// \ingroup AABBTreeGroup
MRMESH_API VertId findDirMax( const Vector3f & dir, const MeshPart & mp, UseAABBTree u = UseAABBTree::Yes );

} //namespace MR
