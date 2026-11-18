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

    /// barycentric coordinates of proj relative to the vertices of proj.face in the order of MeshTopology::getTriVerts( proj.face )
    TriPointf tp;

    /// squared distance from original projected point to proj
    float distSq = 0;

    /// check for validity, otherwise the projection was not found
    [[nodiscard]] bool valid() const { return proj.valid(); }
    [[nodiscard]] explicit operator bool() const { return proj.valid(); }

    /// returns the closest point on mesh as MeshTriPoint relative to topology.edgeWithLeft( proj.face ), or invalid MeshTriPoint if the projection was not found
    [[nodiscard]] MRMESH_API MeshTriPoint mtp( const MeshTopology & topology ) const;
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
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger then the function exits returning upDistLimitSq and no valid point
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
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger then the function exits returning upDistLimitSq and no valid point
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

/**
 * \brief computes the closest point on the triangles of the given tree to given point;
 * the result's tp is relative to the vertices of the triangle in the tree (in the order of tmesh.tris[proj.face] for AABBTree( tmesh ));
 * the coordinates of the triangles are taken from the tree (those of the points it was built or last refit for)
 * \param tree the tree of a mesh or its part, e.g. AABBTree( tmesh, region )
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger then the function exits returning upDistLimitSq and no valid point
 * \param xf mesh-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 * \param validFaces if provided then only faces from there will be considered as projections
 * \param validProjections if provided then only projections passed this test can be returned
 * \param region if provided then only the triangles from there are searched; building the tree only from them is faster
 */
[[nodiscard]] MRMESH_API MeshProjectionResult findProjection( const Vector3f & pt,
    const AABBTree & tree,
    float upDistLimitSq = FLT_MAX,
    const AffineXf3f * xf = nullptr,
    float loDistLimitSq = 0,
    const FacePredicate & validFaces = {},
    const std::function<bool(const MeshProjectionResult&)> & validProjections = {},
    const FaceBitSet * region = nullptr );

/// this callback is invoked on every triangle with bounding box at least partially in the ball (the triangle itself can be fully out of ball),
/// and allows changing (shrinking only) the ball
using FoundBoxedTriCallback = std::function<Processing( FaceId found, Ball3f & ball )>;

/// enumerates all triangles with bounding boxes at least partially in the ball (the triangles themselves can be fully out of ball)
/// until callback returns Stop;
/// the ball during enumeration can shrink (new ball is always within the previous one) but never expand
MRMESH_API void findBoxedTrisInBall( const MeshPart & mp, Ball3f ball, const FoundBoxedTriCallback& foundCallback );

/// enumerates all triangles in the tree (only from region if given) with bounding boxes at least partially in the ball until callback returns Stop;
/// the ball during enumeration can shrink (new ball is always within the previous one) but never expand
MRMESH_API void findBoxedTrisInBall( const AABBTree & tree, Ball3f ball, const FoundBoxedTriCallback& foundCallback, const FaceBitSet * region = nullptr );

/// this callback is invoked on every triangle at least partially in the ball, and allows changing (shrinking only) the ball
using FoundTriCallback = std::function<Processing( const MeshProjectionResult & found, Ball3f & ball )>;

/// enumerates all triangles within the ball until callback returns Stop;
/// the ball during enumeration can shrink (new ball is always within the previous one) but never expand
MRMESH_API void findTrisInBall( const MeshPart & mp, const Ball3f& ball, const FoundTriCallback& foundCallback, const FacePredicate & validFaces = {} );

/// enumerates all triangles in the tree (only from region if given) within the ball until callback returns Stop, points are the coordinates of their vertices;
/// found.tp is relative to the vertices of the triangle in the tree (in the order of tmesh.tris[found.proj.face] for AABBTree( tmesh ));
/// the ball during enumeration can shrink (new ball is always within the previous one) but never expand
MRMESH_API void findTrisInBall( const VertCoords & points, const AABBTree & tree, const Ball3f& ball, const FoundTriCallback& foundCallback, const FacePredicate & validFaces = {},
    const FaceBitSet * region = nullptr );

struct SignedDistanceToMeshResult
{
    /// the closest point on mesh
    PointOnFace proj;
    /// barycentric coordinates of proj relative to the vertices of proj.face in the order of MeshTopology::getTriVerts( proj.face )
    TriPointf tp;
    /// distance from pt to proj (positive - outside, negative - inside the mesh)
    float dist = 0;

    /// returns the closest point on mesh as MeshTriPoint relative to topology.edgeWithLeft( proj.face ), or invalid MeshTriPoint if proj.face is invalid
    [[nodiscard]] MRMESH_API MeshTriPoint mtp( const MeshTopology & topology ) const;
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
