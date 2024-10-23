#pragma once

// distance queries to one mesh only, please see MRMeshMeshDistance.h for queries involving two meshes

#include "MRMeshPart.h"
#include "MRSignDetectionMode.h"
#include <cfloat>
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

struct DistanceToMeshOptions
{
    /// minimum squared distance from a point to mesh
    float minDistSq{ 0 };

    /// maximum squared distance from a point to mesh
    float maxDistSq{ FLT_MAX };

    /// the method to compute distance sign
    SignDetectionMode signMode{ SignDetectionMode::ProjectionNormal };

    /// only for SignDetectionMode::HoleWindingRule:
    /// positive distance if winding number below or equal this threshold;
    /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
    float windingNumberThreshold = 0.5f;

    /// only for SignDetectionMode::HoleWindingRule:
    /// determines the precision of fast approximation: the more the better, minimum value is 1
    float windingNumberBeta = 2;
};

/// computes signed distance from point (p) to mesh part (mp) following options (op);
/// returns std::nullopt if distance is smaller than op.minDist or larger than op.maxDist (except for op.signMode == HoleWindingRule)
[[nodiscard]] MRMESH_API std::optional<float> signedDistanceToMesh( const MeshPart& mp, const Vector3f& p, const DistanceToMeshOptions& op );

/// \}

} // namespace MR
