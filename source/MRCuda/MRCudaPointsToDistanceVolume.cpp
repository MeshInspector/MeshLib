#include "MRCudaPointsToDistanceVolume.h"
#ifndef MRCUDA_NO_VOXELS
#include "MRCudaPointsToDistanceVolume.cuh"

#include "MRCudaBasic.h"
#include "MRCudaMath.h"

#include "MRMesh/MRAABBTreePoints.h"
#include "MRMesh/MRChunkIterator.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRTimer.h"

#include <thread>

#define RETURN_UNEXPECTED( expr ) if ( auto res = ( expr ); !res ) return MR::unexpected( std::move( res.error() ) )

namespace MR
{

namespace Cuda
{

Expected<MR::SimpleVolumeMinMax> pointsToDistanceVolume( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params )
{
    MR_TIMER

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
    const auto bufferSize = maxBufferSizeAlignedByBlock( getCudaSafeMemoryLimit(), params.dimensions, sizeof( float ) );

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

MRCUDA_API Expected<void> pointsToDistanceVolumeByParts( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params,
    std::function<Expected<void> ( const SimpleVolumeMinMax& )> addPart )
{
    MR_TIMER

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
        .origin = fromVec( params.origin ),
        .voxelSize = fromVec( params.voxelSize ),
        .dimensions = fromVec( params.dimensions ),
        .sigma = params.sigma,
        .minWeight = params.minWeight,
    };

    const auto layerSize = (size_t)params.dimensions.x * params.dimensions.y;
    const auto totalSize = layerSize * params.dimensions.z;
    const auto bufferSize = maxBufferSizeAlignedByBlock( getCudaSafeMemoryLimit(), params.dimensions, sizeof( float ) );

    DynamicArrayF cudaVolume;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaVolume.resize( bufferSize ) );

    std::array<MR::SimpleVolumeMinMax, 2> parts;
    for ( auto& part : parts )
    {
        part.dims = params.dimensions;
        part.voxelSize = params.voxelSize;
        part.max = params.sigma * std::exp( -0.5f );
        part.min = -part.max;
    }
    enum Device
    {
        GPU = 0,
        CPU = 1,
    };

    for ( const auto [offset, size] : splitByChunks( totalSize, bufferSize, layerSize ) )
    {
        cudaError_t cudaRes = cudaSuccess;
        auto cudaThread = std::jthread( [&, offset = offset, size = size]
        {
            pointsToDistanceVolumeKernel( cudaNodes.data(), cudaPoints.data(), cudaNormals.data(), cudaVolume.data(), cudaParams, size, offset );
            if ( cudaRes = cudaGetLastError(); cudaRes != cudaSuccess )
                return;

            parts[GPU].dims.z = int( size / layerSize );
            cudaRes = cudaVolume.toVector( parts[GPU].data );
        } );

        // process the previous part during GPU computation
        if ( offset != 0 )
            RETURN_UNEXPECTED( addPart( parts[CPU] ) );

        cudaThread.join();
        if ( cudaRes != cudaSuccess )
            return unexpected( getError( cudaRes ) );

        std::swap( parts[GPU], parts[CPU] );
    }
    // add the last part
    RETURN_UNEXPECTED( addPart( parts[CPU] ) );

    return {};
}

} //namespace Cuda

} //namespace MR
#endif
