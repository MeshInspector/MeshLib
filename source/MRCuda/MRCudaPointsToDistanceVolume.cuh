#include "MRCudaBasic.cuh"
#include "MRCudaMath.cuh"

namespace MR
{
namespace Cuda
{
    // struct similar to MR::PointsToDistanceVolumeParams
    struct PointsToDistanceVolumeParams
    {
        /// origin point of voxels box
        float3 origin;
        /// size of voxel on each axis
        float3 voxelSize;
        /// num voxels along each axis
        int3 dimensions;
        /// it the distance of highest influence of a point;
        /// the maximal influence distance is 3*sigma; beyond that distance the influence is strictly zero
        float sigma = 1;

        /// minimum sum of influence weights from surrounding points for a triangle to appear, meaning that there shall be at least this number of points in close proximity
        float minWeight = 1;
    };
    
    // struct similar to MR::Point
    struct OrderedPoint
    {
        float3 coord;
        int id;
    };

    bool pointsToDistanceVolumeKernel( const Node3* nodes, const OrderedPoint* points, const float3* normals, float* volume, PointsToDistanceVolumeParams params );
}
}