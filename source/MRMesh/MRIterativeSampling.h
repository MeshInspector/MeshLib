#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include <optional>

namespace MR
{

/// performs sampling of cloud points by iteratively removing the point having the closest neighbor in the whole cloud,
/// thus allowing stopping at any given number of samples;
/// returns std::nullopt if it was terminated by the callback
MRMESH_API std::optional<VertBitSet> pointIterativeSampling( const PointCloud& cloud, int numSamples, const ProgressCallback & cb = {} );

} //namespace MR
