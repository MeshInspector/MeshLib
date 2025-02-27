#include "MRCudaPointsToDistanceVolume.h"
#ifndef MRCUDA_NO_VOXELS
#include "MRCudaPointsToDistanceVolume.cuh"

#include "MRCudaBasic.h"

#include "MRMesh/MRAABBTreePoints.h"
#include "MRMesh/MRChunkIterator.h"
#include "MRMesh/MRPointCloud.h"

namespace MR
{

namespace Cuda
{

Expected<MR::SimpleVolumeMinMax> pointsToDistanceVolume( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params )
{
    const auto& tree = cloud.getAABBTree();
    const auto& nodes = tree.nodes();

    DynamicArray<Node3> cudaNodes;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaNodes.fromVector( nodes.vec_ ) );
        
    DynamicArray<OrderedPoint> cudaPoints;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaPoints.fromVector( tree.orderedPoints() ) );

    DynamicArray<float3> cudaNormals;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaNormals.fromVector( params.ptNormals ? params.ptNormals->vec_ : cloud.normals.vec_ ) );

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

    const auto totalSize = (size_t)params.dimensions.x * params.dimensions.y * params.dimensions.z;
    const auto bufferSize = maxBufferSize( getCudaAvailableMemoryForBuffers(), params.dimensions, sizeof( float ) );

    DynamicArrayF cudaVolume;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaVolume.resize( bufferSize ) );

    MR::SimpleVolumeMinMax res;
    res.dims = params.dimensions;
    res.voxelSize = params.voxelSize;
    res.max = params.sigma * std::exp( -0.5f );
    res.min = -res.max;
    res.data.resize( totalSize );

    for ( const auto [offset, size] : splitByChunks( totalSize, bufferSize ) )
    {
        pointsToDistanceVolumeKernel( cudaNodes.data(), cudaPoints.data(), cudaNormals.data(), cudaVolume.data(), cudaParams, size, offset );
        CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );

        CUDA_LOGE_RETURN_UNEXPECTED( cudaVolume.copyTo( res.data.data() + offset, size ) );
    }

    return res;
}

} //namespace Cuda

} //namespace MR
#endif
