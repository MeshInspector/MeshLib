#pragma once

#include "MRMeshFwd.h"
#include <optional>

namespace MR
{

/// computes and returns the distance of traveling from one of start faces to all other reachable faces on the mesh;
/// all unreachable faces will get FLT_MAX value;
/// \param starts all start faces will get value 0 in the result;
/// \param metric metric(e) says the distance of traveling from left(e) to right(e)
/// \param maxDist optional output of the maximal distance to the most distant face
[[nodiscard]] MRMESH_API std::optional<FaceScalars> calcFaceDistances( const MeshTopology & topology, const EdgeMetric & metric, const FaceBitSet & starts,
    float * maxDist = nullptr, const ProgressCallback & cb = {} );

} // namespace MR
