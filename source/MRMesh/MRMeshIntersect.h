#pragma once

#include "MRPointOnFace.h"
#include "MRTriPoint.h"
#include "MRMeshPart.h"
#include "MRMeshTriPoint.h"
#include <cfloat>
#include <functional>

namespace MR
{

struct MeshIntersectionResult
{
    // stores intersected face and global coordinates
    PointOnFace proj;
    // stores barycentric coordinates
    MeshTriPoint mtp;
    // stores the distance from ray origin to the intersection point in direction units
    float distanceAlongLine = 0;
};

// Intersects ray with mesh. Finds the closest to origin intersection
MRMESH_API std::optional<MeshIntersectionResult> rayMeshIntersect( const MeshPart& meshPart, const Line3f& line,
    float rayStart = 0.0f, float rayEnd = FLT_MAX, const IntersectionPrecomputes<float>* prec = nullptr );
// Use double precision
MRMESH_API std::optional<MeshIntersectionResult> rayMeshIntersect( const MeshPart& meshPart, const Line3d& line,
    double rayStart = 0.0, double rayEnd = DBL_MAX, const IntersectionPrecomputes<double>* prec = nullptr );

// this callback is envoked for each encountered ray-mesh intersection;
// if it returns false, then the search immediately terminates
using MeshIntersectionCallback = std::function<bool(const MeshIntersectionResult &)>;
// Intersects ray with mesh. Finds all intersections
MRMESH_API void rayMeshIntersectAll( const MeshPart& meshPart, const Line3f& line, MeshIntersectionCallback callback,
    float rayStart = 0.0f, float rayEnd = FLT_MAX, const IntersectionPrecomputes<float>* prec = nullptr );
// Use double precision
MRMESH_API void rayMeshIntersectAll( const MeshPart& meshPart, const Line3d& line, MeshIntersectionCallback callback,
    double rayStart = 0.0, double rayEnd = DBL_MAX, const IntersectionPrecomputes<double>* prec = nullptr );

} //namespace MR
