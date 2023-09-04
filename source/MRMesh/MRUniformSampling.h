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
MRMESH_API std::optional<VertBitSet> pointUniformSampling( const PointCloud& pointCloud, float distance, const ProgressCallback & cb = {} );

}