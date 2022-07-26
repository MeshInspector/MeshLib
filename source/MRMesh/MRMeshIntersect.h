#pragma once

#include "MRPointOnFace.h"
#include "MRTriPoint.h"
#include "MRMeshPart.h"
#include "MRMeshTriPoint.h"
#include <cfloat>
#include <functional>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

struct MeshIntersectionResult
{
    /// stores intersected face and global coordinates
    PointOnFace proj;
    /// stores barycentric coordinates
    MeshTriPoint mtp;
    /// stores the distance from ray origin to the intersection point in direction units
    float distanceAlongLine = 0;
};

/// Finds ray and mesh intersection in float-precision.
/// \p rayStart and \p rayEnd define the interval on the ray to detect an intersection.
/// \p prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).
/// Finds the closest to ray origin intersection (or any intersection for better performance if \p !closestIntersect).
MRMESH_API std::optional<MeshIntersectionResult> rayMeshIntersect( const MeshPart& meshPart, const Line3f& line,
    float rayStart = 0.0f, float rayEnd = FLT_MAX, const IntersectionPrecomputes<float>* prec = nullptr, bool closestIntersect = true );

/// Finds ray and mesh intersection in double-precision.
/// \p rayStart and \p rayEnd define the interval on the ray to detect an intersection.
/// \p prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).
/// Finds the closest to ray origin intersection (or any intersection for better performance if \p !closestIntersect).
MRMESH_API std::optional<MeshIntersectionResult> rayMeshIntersect( const MeshPart& meshPart, const Line3d& line,
    double rayStart = 0.0, double rayEnd = DBL_MAX, const IntersectionPrecomputes<double>* prec = nullptr, bool closestIntersect = true );

struct MultiMeshIntersectionResult : MeshIntersectionResult
{
    /// the intersection found in this mesh
    const Mesh * mesh = nullptr;
};

template<typename T>
struct Line3Mesh
{
    Line3<T> line; ///< in the reference frame of mesh
    IntersectionPrecomputes<T> * prec = nullptr; ///< set it to a valid pointer for better performance
    const Mesh * mesh = nullptr; ///< must be set a valid pointer before use
    const AABBTree * tree = nullptr; ///< must be set a valid pointer before use
    const FaceBitSet * region = nullptr; ///< may remain nullptr, meaning all mesh
};
using Line3fMesh = Line3Mesh<float>;
using Line3dMesh = Line3Mesh<double>;

/// Intersects ray with many meshes. Finds any intersection (not the closest)
/// \anchor rayMultiMeshAnyIntersectF
MRMESH_API std::optional<MultiMeshIntersectionResult> rayMultiMeshAnyIntersect( const std::vector<Line3fMesh> & lineMeshes,
    float rayStart = 0.0f, float rayEnd = FLT_MAX );
/// Same as \ref rayMultiMeshAnyIntersectF, but use double precision
MRMESH_API std::optional<MultiMeshIntersectionResult> rayMultiMeshAnyIntersect( const std::vector<Line3dMesh> & lineMeshes,
    double rayStart = 0.0, double rayEnd = DBL_MAX );

/// this callback is envoked for each encountered ray-mesh intersection;
/// if it returns false, then the search immediately terminates
using MeshIntersectionCallback = std::function<bool(const MeshIntersectionResult &)>;
/// Intersects ray with mesh. Finds all intersections
/// \anchor rayMeshIntersectAllF
MRMESH_API void rayMeshIntersectAll( const MeshPart& meshPart, const Line3f& line, MeshIntersectionCallback callback,
    float rayStart = 0.0f, float rayEnd = FLT_MAX, const IntersectionPrecomputes<float>* prec = nullptr );
/// Same as \ref rayMeshIntersectAllF, but use double precision
MRMESH_API void rayMeshIntersectAll( const MeshPart& meshPart, const Line3d& line, MeshIntersectionCallback callback,
    double rayStart = 0.0, double rayEnd = DBL_MAX, const IntersectionPrecomputes<double>* prec = nullptr );

/// \}

} // namespace MR
