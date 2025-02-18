#pragma once

#include "MRPointOnFace.h"
#include "MRMeshTriPoint.h"
#include "MRMeshPart.h"
#include "MREnums.h"
#include <cfloat>
#include <optional>
#include <functional>

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

struct MeshProjectionTransforms
{
    const AffineXf3f* rigidXfPoint{ nullptr }; ///< this xf is applied to point to move it into projection space
    const AffineXf3f* nonRigidXfTree{ nullptr }; ///< this xf is applied to AABB tree to move it into projection space
};

/// <summary>
/// Creates structure with simplified transforms for projection functions, with `rigidXfPoint` applied to point, and `nonRigidXfTree` applied to tree
/// </summary>
/// <param name="storageXf">this argument will hold modified transfrom</param>
/// <param name="pointXf">transform for points to be projected</param>
/// <param name="treeXf">transform for tree's boxes</param>
/// <returns>structure with simplified transforms</returns>
MRMESH_API MeshProjectionTransforms createProjectionTransforms( AffineXf3f& storageXf, const AffineXf3f* pointXf, const AffineXf3f* treeXf );

/**
 * \brief computes the closest point on mesh (or its region) to given point
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
 * \param xf mesh-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 * \param validFaces if provided then only faces from there will be considered as projections
 * \param validProjections if provided then only projections passed this test can be returned
 */
[[nodiscard]] MRMESH_API MeshProjectionResult findProjection( const Vector3f & pt, const MeshPart & mp,
    float upDistLimitSq = FLT_MAX,
    const AffineXf3f * xf = nullptr,
    float loDistLimitSq = 0,
    const FacePredicate & validFaces = {},
    const std::function<bool(const MeshProjectionResult&)> & validProjections = {} );

/**
 * \brief computes the closest point on mesh (or its region) to given point
 * \param tree explicitly given BVH-tree for whole mesh or part of mesh we are searching projection on,
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
 * \param xf mesh-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 * \param validFaces if provided then only faces from there will be considered as projections
 * \param validProjections if provided then only projections passed this test can be returned
 */
[[nodiscard]] MRMESH_API MeshProjectionResult findProjectionSubtree( const Vector3f & pt,
    const MeshPart & mp, const AABBTree & tree,
    float upDistLimitSq = FLT_MAX,
    const AffineXf3f * xf = nullptr,
    float loDistLimitSq = 0,
    const FacePredicate & validFaces = {},
    const std::function<bool(const MeshProjectionResult&)> & validProjections = {} );

/// this callback is invoked on every triangle at least partially in the ball, and allows to change the ball
using FoundTriCallback = std::function<Processing( const MeshProjectionResult & found, Ball3f & ball )>;

/// enumerates all triangles within the ball until callback returns Stop;
/// the ball during enumeration can shrink (new ball is always within the previous one) but never expand
MRMESH_API void findTrisInBall( const MeshPart & mp, Ball3f ball, const FoundTriCallback& foundCallback, const FacePredicate & validFaces = {} );

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

/// \}

} // namespace MR
