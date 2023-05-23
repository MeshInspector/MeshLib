#pragma once

#include "MRMeshFwd.h"

namespace MR
{

struct FindOverlappingSettings
{
    /// maximum distance from a triangle point (e.g. centroid) to the nearest triangle
    float maxDistSq = 1e-8f; // suggestion: multiply it on mesh.getBoundingBox().size().lengthSq();
    /// maximum dot product of triangle and its nearest triangle normals
    float maxNormalDot = -0.99f;
};

/// finds all triangles that have oppositely oriented nearest triangle in the mesh
[[nodiscard]] MRMESH_API FaceBitSet findOverlappingTris( const MeshPart & mp, const FindOverlappingSettings & settings );

} //namespace MR
