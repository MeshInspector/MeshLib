#include "MRCudaBasic.cuh"
#include "MRCudaMath.cuh"

namespace MR
{
namespace Cuda
{
    struct PointsToMeshParameters
    {
        /// it the distance of highest influence of a point;
        /// the maximal influence distance is 3*sigma; beyond that distance the influence is strictly zero
        float sigma = 1;

        /// minimum sum of influence weights from surrounding points for a triangle to appear, meaning that there shall be at least this number of points in close proximity
        float minWeight = 1;

        /// optional input: colors of input points
        const uint32_t* ptColors = nullptr;

        /// optional output: averaged colors of mesh vertices
        uint32_t* vColors = nullptr;
    };

    struct SimpleVolume
    {
        DynamicArray<float> data;
        int3 dims;
        float3 voxelSize;
    };

    void pointsToDistanceVolumeKernel( const Node3* nodes, const float3* points, const float3* normals, SimpleVolume* volume, PointsToMeshParameters params );
}
}