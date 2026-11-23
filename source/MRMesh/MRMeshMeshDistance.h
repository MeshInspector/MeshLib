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

/// a point on mesh given both by its position and by its barycentric coordinates
struct PointOnFaceBary
{
    /// the point, and the triangle containing it
    PointOnFace proj;

    /// barycentric coordinates of proj relative to the vertices of proj.face in the order of MeshTopology::getTriVerts( proj.face );
    /// unlike the conversion of proj.point, they place the point exactly on the edge or in the vertex where it was found
    TriPointf tp;

    /// check for validity, otherwise the point was not found
    [[nodiscard]] bool valid() const { return proj.valid(); }
    [[nodiscard]] explicit operator bool() const { return proj.valid(); }

    /// returns the point as MeshTriPoint relative to topology.edgeWithLeft( proj.face ), or invalid MeshTriPoint if the point was not found
    [[nodiscard]] MRMESH_API MeshTriPoint mtp( const MeshTopology & topology ) const;
};

struct MeshMeshDistanceResult
{
    /// two closest points: from meshes A and B respectively
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
    /// two closest points: from meshes A and B respectively, each one as the projection of the other point
    MeshProjectionResult a, b;

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
