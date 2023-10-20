#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include <optional>

namespace MR
{

/// Finds more representative sampling starting from a given one following k-means method
/// returns std::nullopt if it was terminated by the callback
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API std::optional<VertBitSet> improveSampling( const PointCloud & cloud, const VertBitSet & iniSamples, const ProgressCallback & cb = {} );

MRMESH_API bool improveSampling( const PointCloud & cloud, VertBitSet & samples, int numIters, const ProgressCallback & cb = {} );

} //namespace MR
