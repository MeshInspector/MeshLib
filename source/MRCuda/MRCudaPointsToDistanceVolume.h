#include "exports.h"
#include "MRMesh/MRSimpleVolume.h"
#include "MRMesh/MRPointsToDistanceVolume.h"

namespace MR
{

namespace Cuda
{
/// makes SimpleVolume filled with signed distances to points with normals
MRCUDA_API Expected<MR::SimpleVolume> pointsToDistanceVolume( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params );

}
}