#pragma once

#include "MRExpected.h"
#include "MRProgressCallback.h"
#include <cfloat>

namespace MR
{

struct FindOverlappingSettings
{
    /// maximal distance between closest points of one triangle and another overlapping triangle
    float maxDistSq = 1e-10f; // suggestion: multiply it on mesh.getBoundingBox().size().lengthSq();
    
    /// maximal dot product of one triangle and another overlapping triangle normals
    float maxNormalDot = -0.99f;

    /// if the dot product of the normals is at least this value, then similarly oriented close triangles are also considered overlapping,
    /// provided that their projections on the average plane overlap with positive area (it finds thin folds of the surface);
    /// the default value disables this check
    float minNormalDot = FLT_MAX;
    
    /// consider triangle as overlapping only if the area of the oppositely oriented triangle is at least given fraction of the triangle's area
    float minAreaFraction = 1e-5f;
    
    /// if predicate is present it should return true for pair of faces that should be considered as overlap
    std::function<bool( FaceId l, FaceId r )> pred;

    /// for reporting current progress and allowing the user to cancel the algorithm
    ProgressCallback cb;

    // To allow passing Python lambdas into `pred`.
    MR_BIND_PREFER_UNLOCK_GIL_WHEN_USED_AS_PARAM
};

/// finds all triangles that have oppositely (or similarly, see minNormalDot) oriented close triangle in the mesh
[[nodiscard]] MRMESH_API Expected<FaceBitSet> findOverlappingTris( const MeshPart & mp, const FindOverlappingSettings & settings );

} //namespace MR
