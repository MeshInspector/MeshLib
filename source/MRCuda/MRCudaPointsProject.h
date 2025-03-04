#pragma once

#include "exports.h"

#include "MRMesh/MRExpected.h"
#include "MRMesh/MRPointsProject.h"

namespace MR::Cuda
{

// struct from MRPointCloud.cuh
struct PointCloudDataHolder;

/// CUDA-backed implementation of IPointsProjector
class MRCUDA_CLASS PointsProjector : public IPointsProjector
{
public:
    /// sets the reference point cloud
    MRCUDA_API Expected<void> setPointCloud( const PointCloud& pointCloud ) override;

    /// computes the closest points on point cloud to given points
    MRCUDA_API Expected<void> findProjections( std::vector<MR::PointsProjectionResult>& results,
        const std::vector<Vector3f>& points, const FindProjectionOnPointsSettings& settings ) const override;

private:
    std::shared_ptr<PointCloudDataHolder> data_;
};

/// computes the closest points on point cloud to given points
MRCUDA_API Expected<std::vector<MR::PointsProjectionResult>> findProjectionOnPoints( const PointCloud& pointCloud,
    const std::vector<Vector3f>& points, const FindProjectionOnPointsSettings& settings = {} );

/// returns the minimal amount of free GPU memory required for \ref MR::Cuda::findProjectionOnPoints
MRMESH_API size_t findProjectionOnPointsHeapBytes( const PointCloud& pointCloud, size_t pointsCount );

} // namespace MR::Cuda
