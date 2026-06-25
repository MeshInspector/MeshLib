#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

namespace MR
{

/// creates cube's topology with 8 vertices, 12 triangular faces, 18 undirected edges.
/// The order of vertices:
///   0_v: x=min, y=min, z=min
///   1_v: x=min, y=max, z=min
///   2_v: x=max, y=max, z=min
///   3_v: x=max, y=min, z=min
///   4_v: x=min, y=min, z=max
///   5_v: x=min, y=max, z=max
///   6_v: x=max, y=max, z=max
///   7_v: x=max, y=min, z=max
[[nodiscard]] MRMESH_API MeshTopology makeCubeTopology();

/// creates box mesh with given min-corner (base) and given size in every dimension;
/// with default parameters, creates unit cube mesh with the centroid in (0,0,0)
[[nodiscard]] MRMESH_API Mesh makeCube( const Vector3f& size = Vector3f::diagonal(1.0f), const Vector3f& base = Vector3f::diagonal(-0.5f) );

/// creates parallelepiped mesh with given min-corner \p base and given directional vectors \p size
[[nodiscard]] MRMESH_API Mesh makeParallelepiped( const Vector3f side[3], const Vector3f& base );

/// creates mesh visualizing a box
[[nodiscard]] MRMESH_API Mesh makeBoxMesh( const Box3f& box );

} //namespace MR
