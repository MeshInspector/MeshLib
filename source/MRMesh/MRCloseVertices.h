#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// returns a map where each valid vertex is mapped to the smalled valid vertex Id located within given distance (including itself),
/// each vertex not from valid set is mapped to itself
[[nodiscard]] MRMESH_API VertMap findSmallestCloseVertices( const Mesh & mesh, float closeDist );

/// returns a map where each valid vertex is mapped to the smalled valid vertex Id located within given distance (including itself),
/// each vertex not from valid set is mapped to itself
[[nodiscard]] MRMESH_API VertMap findSmallestCloseVertices( const VertCoords & points, const VertBitSet & valid, float closeDist );

} //namespace MR
