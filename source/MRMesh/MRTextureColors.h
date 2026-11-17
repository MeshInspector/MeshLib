#pragma once

#include "MRMeshFwd.h"
#include <optional>

namespace MR
{

/// returns UV-coordinates at the given vertex if they are the same in all surrounding triangles;
/// otherwise returns std::nullopt
[[nodiscard]] MRMESH_API std::optional<UVCoord> findVertexUV( const MeshTopology& topology, VertId v, const TriCornerUVCoords& triCornerUvCoords );

/// if all mesh vertices have unique UV-coordinates in all triangles, then returns them;
/// otherwise returns std::nullopt
[[nodiscard]] MRMESH_API std::optional<VertUVCoords> findVertexUVs( const MeshTopology& topology, const TriCornerUVCoords& triCornerUvCoords );

/// if all vertices from 0 to the maximal vertex in tris have unique UV-coordinates in all triangles, then returns them;
/// otherwise returns std::nullopt
[[nodiscard]] MRMESH_API std::optional<VertUVCoords> findVertexUVs( const Triangulation& tris, const TriCornerUVCoords& triCornerUvCoords );

/// computes the color in the given vertex of mesh textured per-triangle's-corner;
/// if the vertex has different colors in different triangles, then angle-weight average is computed
[[nodiscard]] MRMESH_API Color sampleVertexColor( const Mesh& mesh, VertId v, const MeshTexture& tex, const TriCornerUVCoords& triCornerUvCoords );

/// computes the colors in the vertices of mesh textured per-triangle's-corner;
/// if one vertex has different colors in different triangles, then angle-weight average is computed
[[nodiscard]] MRMESH_API VertColors sampleVertexColors( const Mesh& mesh, const MeshTexture& tex, const TriCornerUVCoords& triCornerUvCoords );

/// computes the colors in the vertices of TriMesh textured per-triangle's-corner;
/// if one vertex has different colors in different triangles, then angle-weight average is computed
[[nodiscard]] MRMESH_API VertColors sampleVertexColors( const TriMesh& mesh, const MeshTexture& tex, const TriCornerUVCoords& triCornerUvCoords );

/// if per-corner UVs can be converted into per-vertex UVs, then puts them in uvCoords and clears triCornerUvCoords;
/// otherwise, if texture is not empty, samples per-vertex colors from it, and drops the texture if uvCoords are empty
MRMESH_API void convertTriCornerUVs( const Mesh& mesh, TriCornerUVCoords& triCornerUvCoords, VertUVCoords& uvCoords, VertColors& colors, MeshTexture& texture );

} //namespace MR
