#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// computes and returns the distance of traveling from one of start faces to all other reachable faces on the mesh;
/// all unreachable faces will get FLT_MAX value;
/// \param starts all start faces will get value 0 in the result;
/// \param metric metric(e) says the distance of traveling from left(e) to right(e)
[[nodiscard]] MRMESH_API FaceScalars calcFaceDistances( const MeshTopology & topology, const EdgeMetric & metric, const FaceBitSet & starts );

} // namespace MR
