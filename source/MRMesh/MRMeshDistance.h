#pragma once

#include "MRPointOnFace.h"
#include "MRMeshPart.h"
#include <cfloat>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

struct MeshDistanceResult
{
    /// two closest points: from meshes A and B respectively
    PointOnFace a, b;
    /// squared distance between a and b
    float distSq = 0;
};

struct MeshSignedDistanceResult
{
    /// two closest points: from meshes A and B respectively
    PointOnFace a, b;
    /// signed distance between a and b, positive if meshes do not collide
    float signedDist = 0;
};

/**
 * \brief computes minimal distance between two meshes or two mesh regions
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid points
 */
MRMESH_API MeshDistanceResult findDistance( const MeshPart & a, const MeshPart & b,
    const AffineXf3f * rigidB2A = nullptr, float upDistLimitSq = FLT_MAX );

/**
 * \brief computes minimal distance between two meshes
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param upDistLimitSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid points
 */
MRMESH_API MeshSignedDistanceResult findSignedDistance( const MeshPart & a, const MeshPart & b,
    const AffineXf3f* rigidB2A = nullptr, float upDistLimitSq = FLT_MAX );

/// \}

} // namespace MR
