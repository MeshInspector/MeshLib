#pragma once

// distance queries involving two meshes, please see MRMeshDistance.h for queries to one mesh only

#include "MRMeshProject.h"
#include "MRFaceFace.h"
#include "MRMeshPart.h"
#include <cfloat>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

struct MeshMeshDistanceResult
{
    /// two closest points: from meshes A and B respectively;
    /// unlike the conversion of a point's position, its barycentric coordinates place it exactly on the edge or in the vertex where it was found
    PointOnFaceBary a, b;
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
    Touching,
    NotColliding ///< the meshes do not touch or collide one another, but inside/outside relation cannot be determined
};

struct MeshMeshSignedDistanceResult
{
    /// two closest points: from meshes A and B respectively
    PointOnFaceBary a, b;

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
 * \param upDistLimitSq upper limit on the squared distance in question, if the real distance is larger then the function exists returning +sqrt(upDistLimitSq) and no valid points
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
 * \brief finds the vertex of B-mesh most distant from A-mesh
 * \return the found vertex in b, its projection on A-mesh in a, and the squared distance between them
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq and no valid point in a
 */
MRMESH_API MeshMeshDistanceResult findMaxDistanceOneWay( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

/**
 * \brief finds the vertex of one mesh most distant from the other mesh (in both directions)
 * \return the found vertex in a or in b, its projection on the other mesh in the remaining field, and the squared distance between them
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq and one valid point only
 */
MRMESH_API MeshMeshDistanceResult findMaxDistance( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

/**
 * \brief returns the maximum of the squared distances from each B-mesh vertex to A-mesh;
 *        please see \ref findHausdorffDistanceOneWay, which considers all points of B-mesh and not only its vertices
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq
 * \sa \ref findMaxDistanceOneWay returning the points as well
 */
MRMESH_API float findMaxDistanceSqOneWay( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

/**
 * \brief returns the maximum of squared distances from each mesh vertex to the other mesh (in both directions);
 *        please see \ref findHausdorffDistance, which considers all points of the meshes and not only their vertices
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq
 * \sa \ref findMaxDistance returning the points as well
 */
MRMESH_API float findMaxDistanceSq( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

/**
 * \brief finds the point of B-mesh (inside its triangles as well, not only in vertices) most distant from A-mesh;
 *        the distance to it is the one-sided Hausdorff distance from B to A
 * \details the maximum can be reached inside a triangle of B, where several triangles of A are equally close,
 *          so the triangles of B are subdivided there till the given tolerance is reached
 * \return the found point in b, its projection on A-mesh in a, and the squared distance between them
 * \param tolerance positive value, the returned distance (not squared) is never above the exact one, and is below it by at most this value;
 *        the smaller the value, the longer the computation; if it is not positive then only the vertices are considered
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq and no valid point in a
 */
MRMESH_API MeshMeshDistanceResult findHausdorffDistanceOneWay( const MeshPart& a, const MeshPart& b, float tolerance, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

/**
 * \brief finds the point of one mesh (not only in vertices) most distant from the other mesh (in both directions);
 *        the distance to it is the Hausdorff distance between two meshes
 * \return the found point in a or in b, its projection on the other mesh in the remaining field, and the squared distance between them
 * \param tolerance positive value, the returned distance (not squared) is never above the exact one, and is below it by at most this value;
 *        the smaller the value, the longer the computation; if it is not positive then only the vertices are considered
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq and one valid point only
 */
MRMESH_API MeshMeshDistanceResult findHausdorffDistance( const MeshPart& a, const MeshPart& b, float tolerance, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

/// \}

} // namespace MR
