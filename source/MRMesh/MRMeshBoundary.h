#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// \defgroup MeshAlgorithmGroup Mesh Algorithm

/// adds triangles along the boundary to straighten it;
/// \details new triangle is added only if 
///  1) aspect ratio of the new triangle is at most maxTriAspectRatio,
///  2) dot product of its normal with neighbor triangles is at least minNeiNormalsDot.
/// \ingroup MeshAlgorithmGroup
MRMESH_API void straightenBoundary( Mesh & mesh, EdgeId bd, float minNeiNormalsDot, float maxTriAspectRatio, FaceBitSet* newFaces = nullptr );

/// returns true if two meshes have exactly the same boundary: every boundary edge of one mesh has a boundary edge in the other mesh
/// with the same coordinates of origin and destination and with the hole on the same side;
/// ids of vertices and edges, and the triangles not touching the boundary can be different
/// \ingroup MeshAlgorithmGroup
[[nodiscard]] MRMESH_API bool sameBoundary( const Mesh & a, const Mesh & b );

} // namespace MR
