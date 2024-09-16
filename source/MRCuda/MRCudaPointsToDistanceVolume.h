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

}
}
#endif