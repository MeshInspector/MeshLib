#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// computes the color in the given vertex of mesh textured per-triangle's-corner;
/// if the vertex has different colors in different triangles, then angle-weight average is computed
[[nodiscard]] MRMESH_API Color sampleVertexColor( const Mesh& mesh, VertId v, const MeshTexture& tex,
    const Triangulation & tris, const TriCornerUVCoords& triCornerUvCoords );

/// computes the colors in the vertices of mesh textured per-triangle's-corner;
/// if one vertex has different colors in different triangles, then angle-weight average is computed
[[nodiscard]] MRMESH_API VertColors sampleVertexColors( const Mesh& mesh, const MeshTexture& tex,
    const Triangulation & tris, const TriCornerUVCoords& triCornerUvCoords );

} //namespace MR
