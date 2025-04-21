#pragma once

#include "MRMeshFwd.h"
#include "MRMinMaxArg.h"

namespace MR
{

/// finds the point having the largest projection on given direction by traversing all region points
[[nodiscard]] MRMESH_API VertId findDirMaxBruteForce( const Vector3f & dir, const VertCoords & points, const VertBitSet * region = nullptr );

/// finds the point having the largest projection on given direction by traversing all region points
[[nodiscard]] MRMESH_API VertId findDirMaxBruteForce( const Vector2f & dir, const VertCoords2 & points, const VertBitSet * region = nullptr );

/// finds the point in the cloud having the largest projection on given direction by traversing all valid points
[[nodiscard]] MRMESH_API VertId findDirMaxBruteForce( const Vector3f & dir, const PointCloud & cloud, const VertBitSet * region = nullptr );

/// finds the vertex in the polyline having the largest projection on given direction by traversing all valid vertices
[[nodiscard]] MRMESH_API VertId findDirMaxBruteForce( const Vector3f & dir, const Polyline3 & polyline );

/// finds the vertex in the polyline having the largest projection on given direction by traversing all valid vertices
[[nodiscard]] MRMESH_API VertId findDirMaxBruteForce( const Vector2f & dir, const Polyline2 & polyline );

/// finds the vertex in the mesh part having the largest projection on given direction by traversing all (region) faces
[[nodiscard]] MRMESH_API VertId findDirMaxBruteForce( const Vector3f & dir, const MeshPart & mp );

/// finds the vertex in the mesh part having the largest projection on given direction by traversing all (region) vertices
[[nodiscard]] MRMESH_API VertId findDirMaxBruteForce( const Vector3f & dir, const MeshVertPart & mp );

/// finds the points having the smallest and the largest projections on given direction by traversing all region points
[[nodiscard]] MRMESH_API MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const VertCoords & points, const VertBitSet * region = nullptr );

/// finds the points having the smallest and the largest projections on given direction by traversing all region points
[[nodiscard]] MRMESH_API MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector2f & dir, const VertCoords2 & points, const VertBitSet * region = nullptr );

/// finds the points in the cloud having the smallest and the largest projections on given direction by traversing all valid points
[[nodiscard]] MRMESH_API MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const PointCloud & cloud, const VertBitSet * region = nullptr );

/// finds the vertex in the polyline having the smallest and the largest projections on given direction by traversing all valid vertices
[[nodiscard]] MRMESH_API MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const Polyline3 & polyline );

/// finds the vertex in the polyline having the smallest and the largest projections on given direction by traversing all valid vertices
[[nodiscard]] MRMESH_API MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector2f & dir, const Polyline2 & polyline );

/// finds the vertices in the mesh part having the smallest and the largest projections on given direction by traversing all (region) faces
[[nodiscard]] MRMESH_API MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const MeshPart & mp );

/// finds the vertices in the mesh part having the smallest and the largest projections on given direction by traversing all (region) vertices
[[nodiscard]] MRMESH_API MinMaxArg<float, VertId> findDirMinMaxBruteForce( const Vector3f & dir, const MeshVertPart & mp );

} //namespace MR
