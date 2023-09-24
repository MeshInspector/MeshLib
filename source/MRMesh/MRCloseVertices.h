#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include <optional>

namespace MR
{

/// returns a map where each valid vertex is mapped to the smallest valid vertex Id located within given distance (including itself), and this smallest vertex is mapped to itself,
/// each vertex not from valid set is mapped to itself
[[nodiscard]] MRMESH_API std::optional<VertMap> findSmallestCloseVertices( const Mesh & mesh, float closeDist, const ProgressCallback & cb = {} );
/// returns a map where each valid vertex is mapped to the smallest valid vertex Id located within given distance (including itself), and this smallest vertex is mapped to itself,
/// each vertex not from valid set is mapped to itself
[[nodiscard]] MRMESH_API std::optional<VertMap> findSmallestCloseVertices( const PointCloud & cloud, float closeDist, const ProgressCallback & cb = {} );
/// returns a map where each valid vertex is mapped to the smallest valid vertex Id located within given distance (including itself), and this smallest vertex is mapped to itself,
/// each vertex not from valid set is mapped to itself; the search tree is constructe inside
[[nodiscard]] MRMESH_API std::optional<VertMap> findSmallestCloseVertices( const VertCoords & points, float closeDist, const VertBitSet * valid = nullptr, const ProgressCallback & cb = {} );
/// returns a map where each valid vertex is mapped to the smallest valid vertex Id located within given distance (including itself), and this smallest vertex is mapped to itself,
/// each vertex not from valid set is mapped to itself; given tree is used as is
[[nodiscard]] MRMESH_API std::optional<VertMap> findSmallestCloseVerticesUsingTree( const VertCoords & points, float closeDist, const AABBTreePoints & tree, const VertBitSet * valid, const ProgressCallback & cb = {} );

/// finds all close vertices, where for each vertex there is another one located within given distance
[[nodiscard]] MRMESH_API std::optional<VertBitSet> findCloseVertices( const Mesh & mesh, float closeDist, const ProgressCallback & cb = {} );
/// finds all close vertices, where for each vertex there is another one located within given distance
[[nodiscard]] MRMESH_API std::optional<VertBitSet> findCloseVertices( const PointCloud & cloud, float closeDist, const ProgressCallback & cb = {} );
/// finds all close vertices, where for each vertex there is another one located within given distance
[[nodiscard]] MRMESH_API std::optional<VertBitSet> findCloseVertices( const VertCoords & points, float closeDist, const VertBitSet * valid = nullptr, const ProgressCallback & cb = {} );
/// finds all close vertices, where for each vertex there is another one located within given distance; smallestMap is the result of findSmallestCloseVertices function call
[[nodiscard]] MRMESH_API VertBitSet findCloseVertices( const VertMap & smallestMap );

// definition: if A,B and C,D are close vertices, then let us name AC and BD twin edges

/// finds a mapping from an edge to a twin edge in given mesh (each twin edge will be present either in key or in the value of returned map)
[[nodiscard]] MRMESH_API EdgeHashMap findTwinEdgeHashMap( const Mesh & mesh, float closeDist );
/// finds all twin edges in the mesh
[[nodiscard]] MRMESH_API EdgeBitSet findTwinEdges( const Mesh & mesh, float closeDist );
/// finds all twin edges from given map
[[nodiscard]] MRMESH_API EdgeBitSet findTwinEdges( const EdgeHashMap & emap );
/// finds all twin edges in the mesh
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet findTwinUndirectedEdges( const Mesh & mesh, float closeDist );
/// finds all twin edges from given map
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet findTwinUndirectedEdges( const EdgeHashMap & emap );

} //namespace MR
