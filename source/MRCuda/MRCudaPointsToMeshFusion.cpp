#include "MRCudaPointsToMeshFusion.h"
#include "MRCudaPointsToMeshFusion.cuh"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRAABBTreePoints.h"

namespace MR
{
namespace Cuda
{
    Expected<MR::SimpleVolume> pointsToDistanceVolume( const PointCloud& cloud, const MR::PointsToMeshParameters& params )
    {
        const auto& tree = cloud.getAABBTree();
        const auto& nodes = tree.nodes();

        DynamicArray<Node3> cudaNodes;
        cudaNodes.fromVector( nodes.vec_ );
        
        DynamicArray<float3> cudaPoints;
        cudaPoints.fromVector( cloud.points.vec_ );

        DynamicArray<float3> cudaNormals;
        cudaNormals.fromVector( cloud.normals.vec_ );

        PointsToMeshParameters cudaParams
        {
            .sigma = params.sigma,
            .minWeight = params.minWeight
        };

        SimpleVolume cudaVolume;
        pointsToDistanceVolumeKernel( cudaNodes.data(), cudaPoints.data(), cudaNormals.data(), &cudaVolume, cudaParams );

        MR::SimpleVolume res;
        res.dims = { cudaVolume.dims.x, cudaVolume.dims.y, cudaVolume.dims.z };
        res.voxelSize = { cudaVolume.voxelSize.x, cudaVolume.voxelSize.y, cudaVolume.voxelSize.z };
        cudaVolume.data.toVector( res.data );
        return res;
    }
}
}