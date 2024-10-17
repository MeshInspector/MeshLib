#include "MRCudaPointsToDistanceVolume.h"
#ifndef MRCUDA_NO_VOXELS
#include "MRCudaPointsToDistanceVolume.cuh"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRAABBTreePoints.h"

namespace MR
{

namespace Cuda
{

Expected<MR::SimpleVolumeMinMax> pointsToDistanceVolume( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params )
{
    const auto& tree = cloud.getAABBTree();
    const auto& nodes = tree.nodes();

    DynamicArray<Node3> cudaNodes;
    cudaNodes.fromVector( nodes.vec_ );
        
    DynamicArray<OrderedPoint> cudaPoints;
    cudaPoints.fromVector( tree.orderedPoints() );

    DynamicArray<float3> cudaNormals;
    cudaNormals.fromVector( params.ptNormals ? params.ptNormals->vec_ : cloud.normals.vec_ );

    PointsToDistanceVolumeParams cudaParams
    {
        .sigma = params.sigma,
        .minWeight = params.minWeight
    };

    cudaParams.origin.x = params.origin.x;
    cudaParams.origin.y = params.origin.y;
    cudaParams.origin.z = params.origin.z;

    cudaParams.voxelSize.x = params.voxelSize.x;
    cudaParams.voxelSize.y = params.voxelSize.y;
    cudaParams.voxelSize.z = params.voxelSize.z;

    cudaParams.dimensions.x = params.dimensions.x;
    cudaParams.dimensions.y = params.dimensions.y;
    cudaParams.dimensions.z = params.dimensions.z;
       
    DynamicArray<float> cudaVolume;
    cudaVolume.resize( size_t( params.dimensions.x ) * params.dimensions.y * params.dimensions.z );
    if ( !pointsToDistanceVolumeKernel( cudaNodes.data(), cudaPoints.data(), cudaNormals.data(), cudaVolume.data(), cudaParams ) )
        return unexpected( "CUDA error occurred" );

    MR::SimpleVolumeMinMax res;
    res.dims = params.dimensions;
    res.voxelSize = params.voxelSize;
    res.max = params.sigma * std::exp( -0.5f );
    res.min = -res.max;
    cudaVolume.toVector( res.data );
    return res;
}

} //namespace Cuda

} //namespace MR
#endif
