#pragma once

#include "MRMeshFwd.h"
#include "MREnums.h"

namespace MR
{

/// finds the vertex in the mesh part having the largest projection on given direction,
/// optionally uses aabb-tree inside for faster computation
/// \ingroup AABBTreeGroup
[[nodiscard]] MRMESH_API VertId findDirMax( const Vector3f & dir, const Mesh & m, UseAABBTree u = UseAABBTree::Yes );

/// finds the vertex in the mesh part having the largest projection on given direction,
/// optionally uses aabb-tree inside for faster computation
/// \ingroup AABBTreeGroup
[[nodiscard]] MRMESH_API VertId findDirMax( const Vector3f & dir, const MeshPart & mp, UseAABBTree u = UseAABBTree::Yes );

/// finds the vertex in the mesh part having the largest projection on given direction,
/// optionally uses aabb-points-tree inside for faster computation
/// \ingroup AABBTreeGroup
[[nodiscard]] MRMESH_API VertId findDirMax( const Vector3f & dir, const MeshVertPart & mp, UseAABBTree u = UseAABBTree::Yes );

/// finds the vertex in the polyline having the largest projection on given direction,
/// optionally uses aabb-tree inside for faster computation
/// \ingroup AABBTreeGroup
[[nodiscard]] MRMESH_API VertId findDirMax( const Vector3f & dir, const Polyline3 & polyline, UseAABBTree u = UseAABBTree::Yes );

/// finds the vertex in the polyline having the largest projection on given direction,
/// optionally uses aabb-tree inside for faster computation
/// \ingroup AABBTreeGroup
[[nodiscard]] MRMESH_API VertId findDirMax( const Vector2f & dir, const Polyline2 & polyline, UseAABBTree u = UseAABBTree::Yes );

/// finds the point in the cloud having the largest projection on given direction,
/// optionally uses aabb-tree inside for faster computation
/// \ingroup AABBTreeGroup
[[nodiscard]] MRMESH_API VertId findDirMax( const Vector3f & dir, const PointCloud & cloud, const VertBitSet * region = nullptr, UseAABBTree u = UseAABBTree::Yes );

/// finds the point in the tree having the largest projection on given direction
/// \ingroup AABBTreeGroup
[[nodiscard]] MRMESH_API VertId findDirMax( const Vector3f & dir, const AABBTreePoints & tree, const VertBitSet * region = nullptr );

} //namespace MR

