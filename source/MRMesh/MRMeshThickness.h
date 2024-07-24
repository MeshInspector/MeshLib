#pragma once

#include "MRMeshFwd.h"
#include <optional>

namespace MR
{

/// returns the distance from each vertex along minus normal to the nearest mesh intersection (or FLT_MAX if no intersection found)
[[nodiscard]] MRMESH_API VertScalars computeRayThicknessAtVertices( const Mesh& mesh );
[[deprecated]] MRMESH_API VertScalars computeThicknessAtVertices( const Mesh& mesh );

/// returns the nearest intersection between the mesh and the ray from given point along minus normal (inside the mesh)
[[nodiscard]] MRMESH_API std::optional<MeshIntersectionResult> rayInsideIntersect( const Mesh& mesh, const MeshTriPoint & p );
[[nodiscard]] MRMESH_API std::optional<MeshIntersectionResult> rayInsideIntersect( const Mesh& mesh, VertId v );

} // namespace MR
