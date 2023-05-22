#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// returns a map where each valid vertex is mapped to the smallest valid vertex Id located within given distance (including itself), and this smallest vertex is mapped to itself,
/// each vertex not from valid set is mapped to itself
[[nodiscard]] MRMESH_API VertMap findSmallestCloseVertices( const Mesh & mesh, float closeDist );

/// returns a map where each valid vertex is mapped to the smallest valid vertex Id located within given distance (including itself), and this smallest vertex is mapped to itself,
/// each vertex not from valid set is mapped to itself
[[nodiscard]] MRMESH_API VertMap findSmallestCloseVertices( const VertCoords & points, float closeDist, const VertBitSet * valid = nullptr );

} //namespace MR
