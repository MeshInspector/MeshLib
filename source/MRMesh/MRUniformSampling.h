#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRProgressCallback.h"
#include <optional>

namespace MR
{

/// Sample vertices, removing ones that are too close;
/// returns std::nullopt if it was terminated by the callback
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API std::optional<VertBitSet> pointUniformSampling( const PointCloud& pointCloud, float distance, const ProgressCallback & cb = {} );

/// Composes new point cloud consisting of uniform samples of original point cloud;
/// returns std::nullopt if it was terminated by the callback
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API std::optional<PointCloud> makeUniformSampledCloud( const PointCloud& pointCloud, float distance, const ProgressCallback & cb = {} );

} //namespace MR
