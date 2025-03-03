#pragma once

#include "exports.h"

#include "MRMesh/MRExpected.h"
#include "MRMesh/MRPointsProject.h"

namespace MR::Cuda
{

// struct from MRPointCloud.cuh
struct PointCloudDataHolder;

/// ...
class PointsProjector : public IPointsProjector
{
public:
    /// ...
    MRCUDA_API Expected<void> setPointCloud( const PointCloud& pointCloud ) override;

    /// ...
    [[nodiscard]] MRCUDA_API Expected<std::vector<MR::PointsProjectionResult>> findProjections(
        const std::vector<Vector3f>& points, const FindProjectionOnPointsSettings& settings ) const override;

private:
    std::unique_ptr<PointCloudDataHolder> data_;
};

/// computes the closest points on point cloud to given points
MRCUDA_API Expected<std::vector<MR::PointsProjectionResult>> findProjectionOnPoints( const PointCloud& pointCloud,
    const std::vector<Vector3f>& points, const FindProjectionOnPointsSettings& settings = {} );

/// returns the minimal amount of free GPU memory required for \ref MR::Cuda::findProjectionOnPoints
MRMESH_API size_t findProjectionOnPointsHeapBytes( const PointCloud& pointCloud, size_t pointsCount );

} // namespace MR::Cuda
