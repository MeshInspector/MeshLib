#pragma once

#include "config.h"
#ifndef MRCUDA_NO_VOXELS
#include "exports.h"
#include "MRVoxels/MRVoxelsVolume.h"
#include "MRVoxels/MRPointsToDistanceVolume.h"

namespace MR
{

namespace Cuda
{
/// makes SimpleVolume filled with signed distances to points with normals
MRCUDA_API Expected<MR::SimpleVolumeMinMax> pointsToDistanceVolume( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params );

/// makes SimpleVolume filled with signed distances to points with normals
/// populate the volume by parts to the given callback
MRCUDA_API Expected<void> pointsToDistanceVolumeByParts( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params,
    std::function<Expected<void> ( const SimpleVolumeMinMax& volume, int zOffset )> addPart, int layerOverlap );

/// returns the minimal amount of free GPU memory required to build a distance volume with given dimensions
/// \param ptNormals (optional) point normals to be used instead of the ones stored in the cloud
MRCUDA_API size_t pointsToDistanceVolumeMemory( const PointCloud& cloud, const Vector3i& dims, const VertNormals* ptNormals );

/// CUDA implementation of IComputePointsToDistanceVolume
class MRCUDA_CLASS ComputePointsToDistanceVolume : public MR::IComputePointsToDistanceVolume
{
public:
    // see methods' descriptions in MR::IComputePointsToDistanceVolume
    MRCUDA_API bool canCompute( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params ) const override;
    MRCUDA_API Expected<MR::SimpleVolumeMinMax> compute( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params ) const override;
    MRCUDA_API bool supportsByParts() const override;
    MRCUDA_API Expected<void> computeByParts( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params,
        AddPartFunc addPart, int layerOverlap ) const override;
};

}
}
#endif
