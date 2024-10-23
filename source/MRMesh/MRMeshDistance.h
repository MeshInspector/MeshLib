#pragma once

#include "MRPointOnFace.h"
#include "MRMeshPart.h"
#include <cfloat>
#include <functional>

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
using MeshDistanceResult [[deprecated]] = MeshMeshDistanceResult;

struct MeshMeshSignedDistanceResult
{
    /// two closest points: from meshes A and B respectively
    PointOnFace a, b;
    /// signed distance between a and b, positive if meshes do not collide
    float signedDist = 0;
};
using MeshSignedDistanceResult [[deprecated]] = MeshMeshSignedDistanceResult;

/**
 * \brief computes minimal distance between two meshes or two mesh regions
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid points
 */
MRMESH_API MeshMeshDistanceResult findDistance( const MeshPart & a, const MeshPart & b,
    const AffineXf3f * rigidB2A = nullptr, float upDistLimitSq = FLT_MAX );

/**
 * \brief computes minimal distance between two meshes
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param upDistLimitSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid points
 */
MRMESH_API MeshMeshSignedDistanceResult findSignedDistance( const MeshPart & a, const MeshPart & b,
    const AffineXf3f* rigidB2A = nullptr, float upDistLimitSq = FLT_MAX );

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

enum class ProcessOneResult : bool
{
    StopProcessing = false,
    ContinueProcessing = true
};

/// this callback is invoked for every triangle in range, where
/// \param p closest point on original triangle
/// \param f triangle id in question
/// \param q closest point on f-triangle
/// \param distSq squared distance in between p and q
/// \return whether to continue or to stop processing other triangles
using TriangleCallback = std::function<ProcessOneResult( const Vector3f & p, FaceId f, const Vector3f & q, float distSq )>;

/// invokes given callback for all triangles from given mesh part located not further than
/// given squared distance from t-triangle
MRMESH_API void processCloseTriangles( const MeshPart& mp, const Triangle3f & t, float rangeSq, const TriangleCallback & call );

/// \}

} // namespace MR
