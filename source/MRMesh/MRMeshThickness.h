#pragma once

#include "MRMeshFwd.h"
#include "MRMeshProject.h"
#include <optional>

namespace MR
{

/// returns the distance from each vertex along minus normal to the nearest mesh intersection (or FLT_MAX if no intersection found)
[[nodiscard]] MRMESH_API VertScalars computeRayThicknessAtVertices( const Mesh& mesh );
[[deprecated]] MRMESH_API VertScalars computeThicknessAtVertices( const Mesh& mesh );

/// describes the point of measurement on mesh
struct MeshPoint
{
    MeshTriPoint triPoint; ///< relative position on mesh
    Vector3f pt;           ///< 3d coordinates
    Vector3f inDir;        ///< unit direction inside the mesh = minus normal

    MRMESH_API void set( const Mesh& mesh, const MeshTriPoint & p );
};

/// returns the nearest intersection between the mesh and the ray from given point along minus normal (inside the mesh)
[[nodiscard]] MRMESH_API std::optional<MeshIntersectionResult> rayInsideIntersect( const Mesh& mesh, const MeshPoint & m );
[[nodiscard]] MRMESH_API std::optional<MeshIntersectionResult> rayInsideIntersect( const Mesh& mesh, VertId v );

/// Shrinking sphere: A parallel algorithm for computing the thickness of 3D objects
/// https://www.cad-journal.net/files/vol_13/CAD_13(2)_2016_199-207.pdf

/// controls the finding of inscribed sphere in mesh
struct InSphereSearchSettings
{
    /// maximum allowed radius of the sphere
    float maxRadius = 1;

    /// maximum number of shrinking iterations
    int maxIter = 8;
};

/// found inscribed sphere touching input point with center along given direction
struct InSphere
{
    Vector3f center;
    float radius = 0;
    MeshProjectionResult oppositeTouchPoint; ///< not input point and on incident triangles
};

/// finds sphere inscribed in the mesh touching point (p) with center along the normal at (p)
[[nodiscard]] MRMESH_API InSphere findInCircle( const Mesh& mesh, const MeshPoint & m, const InSphereSearchSettings & settings );

} // namespace MR
