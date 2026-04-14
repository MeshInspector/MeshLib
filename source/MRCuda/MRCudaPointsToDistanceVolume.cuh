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

        /// coefficient used for weight calculation: e^(dist^2 * -invSigmaModifier * sigma^-2)
        /// values: (0;inf)
        float invSigmaModifier = 0.5f;

        /// changes the way point angle affects weight, by default it is linearly increasing with dot product
        /// if enabled - increasing as dot product^(0.5) (with respect to its sign)
        bool sqrtAngleWeight{ false };
    };
    
    // struct similar to MR::Point
    struct OrderedPoint
    {
        float3 coord;
        int id;
    };

    void pointsToDistanceVolumeKernel( const Node3* nodes, const OrderedPoint* points, const float3* normals, float* volume, PointsToDistanceVolumeParams params, size_t chunkSize, size_t chunkOffset );
}
}