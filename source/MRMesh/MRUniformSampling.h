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

/// Sample vertices, removing ones that are too close;
/// may take longer than MR::pointUniformSampling but return more regular distribution
/// returns std::nullopt if it was terminated by the callback
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API std::optional<VertBitSet> pointRegularUniformSampling( const PointCloud& pointCloud, float distance, const ProgressCallback& cb = {} );

/// Composes new point cloud consisting of uniform samples of original point cloud;
/// \param extNormals if given then they will be copied in new point cloud
/// returns std::nullopt if it was terminated by the callback
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API std::optional<PointCloud> makeUniformSampledCloud( const PointCloud& pointCloud, float distance,
    const VertNormals * extNormals = nullptr, const ProgressCallback & cb = {} );

} //namespace MR
