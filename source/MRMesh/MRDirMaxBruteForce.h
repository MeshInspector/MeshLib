#pragma once

#include "MRMeshFwd.h"
#include "MRMinMaxArg.h"

namespace MR
{

/// finds the point having the largest projection on given direction by traversing all region points
[[nodiscard]] MRMESH_API VertId findDirMaxBruteForce( const Vector3f & dir, const VertCoords & points, const VertBitSet * region = nullptr );

/// finds the point in the cloud having the largest projection on given direction by traversing all valid points
[[nodiscard]] MRMESH_API VertId findDirMaxBruteForce( const Vector3f & dir, const PointCloud & cloud );

/// finds the vertex in the polyline having the largest projection on given direction by traversing all valid vertices
[[nodiscard]] MRMESH_API VertId findDirMaxBruteForce( const Vector3f & dir, const Polyline3 & polyline );

/// finds the vertex in the mesh part having the largest projection on given direction by traversing all (region) faces
[[nodiscard]] MRMESH_API VertId findDirMaxBruteForce( const Vector3f & dir, const MeshPart & mp );

/// finds the points having the smallest and the largest projections on given direction by traversing all region points
[[nodiscard]] MRMESH_API MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const VertCoords & points, const VertBitSet * region = nullptr );

/// finds the points in the cloud having the smallest and the largest projections on given direction by traversing all valid points
[[nodiscard]] MRMESH_API MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const PointCloud & cloud );

/// finds the vertex in the polyline having the smallest and the largest projections on given direction by traversing all valid vertices
[[nodiscard]] MRMESH_API MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const Polyline3 & polyline );

/// finds the vertices in the mesh part having the smallest and the largest projections on given direction by traversing all (region) faces
[[nodiscard]] MRMESH_API MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const MeshPart & mp );

} //namespace MR
