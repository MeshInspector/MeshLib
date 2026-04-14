#pragma once

// distance queries involving two meshes, please see MRMeshDistance.h for queries to one mesh only

#include "MRPointOnFace.h"
#include "MRFaceFace.h"
#include "MRMeshPart.h"
#include <cfloat>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

struct MeshMeshDistanceResult
{
    /// two closest points: from meshes A and B respectively
    PointOnFace a, b;
    /// squared distance between a and b
    float distSq = 0;
};

enum class MeshMeshCollisionStatus
{
    BothOutside,
    BothInside,
    AInside,
    BInside,
    Colliding,
    Touching
};

struct MeshMeshSignedDistanceResult
{
    /// two closest points: from meshes A and B respectively
    PointOnFace a, b;

    /// mutual status of two meshes
    MeshMeshCollisionStatus status{ MeshMeshCollisionStatus::BothOutside };

    /// signed distance between a and b, positive if meshes do not collide
    float signedDist = 0;
};

/**
 * \brief computes minimal distance between two meshes or two mesh regions
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger then the function exists returning upDistLimitSq and no valid points
  */
MRMESH_API MeshMeshDistanceResult findDistance( const MeshPart & a, const MeshPart & b,
    const AffineXf3f * rigidB2A = nullptr, float upDistLimitSq = FLT_MAX );

/**
 * \brief computes minimal distance between two meshes
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param upDistLimitSq upper limit on the positive distance in question, if the real distance is larger then the function exists returning upDistLimitSq and no valid points
 * \note if one mesh is fully inside the other one - closest points are returned
 */
MRMESH_API MeshMeshSignedDistanceResult findSignedDistance( const MeshPart & a, const MeshPart & b,
    const AffineXf3f* rigidB2A = nullptr, float upDistLimitSq = FLT_MAX );

/**
 * \brief finds if two meshes are touching, colliding or inside each other
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 */
MRMESH_API MeshMeshCollisionStatus findCollisionStatus( const MeshPart& a, const MeshPart& b,
    const AffineXf3f* rigidB2A = nullptr );

/**
 * \brief finds if two meshes are touching, colliding or inside each other
 * \param distRes result of findDistance on these input
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param collisions optional output collisions among A and B
 */
MRMESH_API MeshMeshCollisionStatus findCollisionStatus( const MeshPart& a, const MeshPart& b, 
    const MeshMeshDistanceResult& distRes, const AffineXf3f* rigidB2A = nullptr, std::vector<FaceFace>* collisions = nullptr );

/**
 * \brief returns the maximum of the squared distances from each B-mesh vertex to A-mesh
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq
 */
MRMESH_API float findMaxDistanceSqOneWay( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

/**
 * \brief returns the squared Hausdorff distance between two meshes, that is
          the maximum of squared distances from each mesh vertex to the other mesh (in both directions)
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq
 */
MRMESH_API float findMaxDistanceSq( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

/// \}

} // namespace MR
