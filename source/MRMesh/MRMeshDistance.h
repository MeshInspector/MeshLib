#pragma once

// distance queries to one mesh only, please see MRMeshMeshDistance.h for queries involving two meshes

#include "MRMeshPart.h"
#include <functional>

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

/// \}

} // namespace MR
