#pragma once

#include "MRPointOnFace.h"
#include "MRMeshTriPoint.h"
#include "MRMeshPart.h"
#include <cfloat>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

struct MeshProjectionResult
{
    /// the closest point on mesh, transformed by xf if it is given
    PointOnFace proj;
    /// its barycentric representation
    MeshTriPoint mtp;
    /// squared distance from pt to proj
    float distSq = 0;
};

/// 
/**
 * \brief computes the closest point on mesh (or its region) to given point
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
 * \param xf mesh-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 * \param skipFace this triangle will be skipped and never returned as a projection
 */
[[nodiscard]] MRMESH_API MeshProjectionResult findProjection( const Vector3f & pt, const MeshPart & mp,
    float upDistLimitSq = FLT_MAX,
    const AffineXf3f * xf = nullptr,
    float loDistLimitSq = 0,
    FaceId skipFace = {} );

struct SignedDistanceToMeshResult
{
    /// the closest point on mesh
    PointOnFace proj;
    /// its barycentric representation
    MeshTriPoint mtp;
    /// distance from pt to proj (positive - outside, negative - inside the mesh)
    float dist = 0;
};

/**
 * \brief computes the closest point on mesh (or its region) to given point,
 * and finds the distance with sign to it (positive - outside, negative - inside the mesh)
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger then the function exits returning nullopt
 * \param loDistLimitSq low limit on the distance in question, if the real distance smaller then the function exits returning nullopt
 */
[[nodiscard]] MRMESH_API std::optional<SignedDistanceToMeshResult> findSignedDistance( const Vector3f & pt, const MeshPart & mp,
    float upDistLimitSq = FLT_MAX, float loDistLimitSq = 0 );

/// Finds inner-shell vertices on bidirectional \param shell constructed for an open \param mesh;
/// \param side - positive to find outer vertices, negative to find inner vertices
/// The function will return all shell vertices that have distance to mesh of same sign as \param side
/// excluding the vertices projecting on mesh boundary
[[nodiscard]] MRMESH_API VertBitSet findInnerShellVerts( const Mesh & mesh, const Mesh & shell, float side );

/// \}

} // namespace MR
