#include "exports.h"
#include "MRMesh/MRSimpleVolume.h"
#include "MRMesh/MRPointsToDistanceVolume.h"

namespace MR
{

namespace Cuda
{

MRCUDA_API Expected<MR::SimpleVolume> pointsToDistanceVolume( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params );

}
}