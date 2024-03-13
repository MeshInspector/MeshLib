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
        
        DynamicArray<OrderedPoint> cudaPoints;
        cudaPoints.fromVector( tree.orderedPoints() );

        DynamicArray<float3> cudaNormals;
        cudaNormals.fromVector( cloud.normals.vec_ );

        PointsToMeshParameters cudaParams
        {
            .sigma = params.sigma,
            .minWeight = params.minWeight
        };

        auto box = cloud.getBoundingBox();
        auto expansion = Vector3f::diagonal( 2 * params.voxelSize );
       
        SimpleVolume cudaVolume;
        const auto origin = box.min - expansion;
        cudaVolume.origin.x = origin.x;
        cudaVolume.origin.y = origin.y;
        cudaVolume.origin.z = origin.z;

        cudaVolume.voxelSize.x = cudaVolume.voxelSize.y = cudaVolume.voxelSize.z = params.voxelSize;
        const auto dimensions = Vector3i( ( box.max + expansion - origin ) / params.voxelSize ) + Vector3i::diagonal( 1 );
        cudaVolume.dims.x = dimensions.x;
        cudaVolume.dims.y = dimensions.y;
        cudaVolume.dims.z = dimensions.z;
        cudaVolume.data.resize( cudaVolume.dims.x * cudaVolume.dims.y * cudaVolume.dims.z );

        pointsToDistanceVolumeKernel( cudaNodes.data(), cudaPoints.data(), cudaNormals.data(), &cudaVolume, cudaParams );

        MR::SimpleVolume res;
        res.dims = { cudaVolume.dims.x, cudaVolume.dims.y, cudaVolume.dims.z };
        res.voxelSize = { cudaVolume.voxelSize.x, cudaVolume.voxelSize.y, cudaVolume.voxelSize.z };
        res.max = params.sigma * std::exp( -0.5f );
        res.min = -res.max;
        cudaVolume.data.toVector( res.data );
        return res;
    }
}
}