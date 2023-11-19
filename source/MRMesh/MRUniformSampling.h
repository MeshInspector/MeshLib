#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRProgressCallback.h"
#include <optional>

namespace MR
{

struct UniformSamplingSettings
{
    /// minimal distance between samples
    float distance = 0;
    /// if point cloud has normals then automatically decreases local distance to make sure that all points inside have absolute normal dot product not less than this value;
    /// this is to make sampling denser in the regions of high curvature;
    /// value <=0 means ignore normals;
    /// value >=1 means select all points (practically useless)
    float minNormalDot = 0;
    /// if true process the points in lexicographical order, which gives tighter and more uniform samples;
    /// if false process the points according to their ids, which is faster
    bool lexicographicalOrder = true;
    /// if not nullptr then these normals will be used during sampling instead of normals in the cloud itself
    const VertNormals * pNormals = nullptr;
    /// to report progress and cancel processing
    ProgressCallback progress;
};

/// Sample vertices, removing ones that are too close;
/// returns std::nullopt if it was terminated by the callback
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API std::optional<VertBitSet> pointUniformSampling( const PointCloud& pointCloud, const UniformSamplingSettings & settings );


/// Composes new point cloud consisting of uniform samples of original point cloud;
/// returns std::nullopt if it was terminated by the callback
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API std::optional<PointCloud> makeUniformSampledCloud( const PointCloud& pointCloud, const UniformSamplingSettings & settings );

} //namespace MR
