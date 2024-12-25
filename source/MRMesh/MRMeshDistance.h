#pragma once

// distance queries to one mesh only, please see MRMeshMeshDistance.h for queries involving two meshes

#include "MRMeshPart.h"
#include "MRDistanceToMeshOptions.h"
#include <functional>
#include <optional>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

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

/// computes signed distance from point (p) to mesh part (mp) following options (op);
/// returns std::nullopt if distance is smaller than op.minDist or larger than op.maxDist (except for op.signMode == HoleWindingRule)
[[nodiscard]] MRMESH_API std::optional<float> signedDistanceToMesh( const MeshPart& mp, const Vector3f& p, const SignedDistanceToMeshOptions& op );

/// \}

} // namespace MR
