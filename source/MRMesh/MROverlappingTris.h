#pragma once

#include "MRExpected.h"
#include "MRProgressCallback.h"

namespace MR
{

struct FindOverlappingSettings
{
    /// maximal distance between closest points of one triangle and another overlapping triangle
    float maxDistSq = 1e-10f; // suggestion: multiply it on mesh.getBoundingBox().size().lengthSq();
    /// maximal dot product of one triangle and another overlapping triangle normals
    float maxNormalDot = -0.99f;
    /// consider triangle as overlapping only if the area of the oppositely oriented triangle is at least given fraction of the triangle's area
    float minAreaFraction = 1e-5f;
    /// for reporting current progress and allowing the user to cancel the algorithm
    ProgressCallback cb;
};

/// finds all triangles that have oppositely oriented close triangle in the mesh
[[nodiscard]] MRMESH_API Expected<FaceBitSet> findOverlappingTris( const MeshPart & mp, const FindOverlappingSettings & settings );

} //namespace MR
