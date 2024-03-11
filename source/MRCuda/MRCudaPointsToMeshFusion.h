#include "exports.h"
#include "MRMesh/MRSimpleVolume.h"
#include "MRMesh/MRPointsToMeshFusion.h"

namespace MR
{

namespace Cuda
{

MRCUDA_API Expected<MR::SimpleVolume> pointsToDistanceVolume( const PointCloud& cloud, const MR::PointsToMeshParameters& params );

}
}