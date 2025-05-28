#include "MRCudaPointsToDistanceVolume.h"
#ifndef MRCUDA_NO_VOXELS
#include "MRCudaPointsToDistanceVolume.cuh"

#include "MRCudaBasic.h"
#include "MRCudaMath.h"
#include "MRCudaPipeline.h"

#include "MRMesh/MRAABBTreePoints.h"
#include "MRMesh/MRChunkIterator.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRTimer.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

namespace Cuda
{

Expected<MR::SimpleVolumeMinMax> pointsToDistanceVolume( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params )
{
    MR_TIMER;

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
    std::function<Expected<void> ( const SimpleVolumeMinMax&, int )> addPart, int layerOverlap )
{
    MR_TIMER;

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
    if ( bufferSize != totalSize )
    {
        spdlog::debug( "Not enough free GPU memory to process all data at once; processing in several iterations" );
        spdlog::debug(
            "Required memory: {}, available memory: {}, iterations: {}",
            bytesString( totalSize * sizeof( float ) ),
            bytesString( bufferSize * sizeof( float ) ),
            chunkCount( totalSize, bufferSize )
        );
    }

    DynamicArrayF cudaVolume;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaVolume.resize( bufferSize ) );

    MR::SimpleVolumeMinMax part;
    part.dims = params.dimensions;
    part.voxelSize = params.voxelSize;
    part.max = params.sigma * std::exp( -0.5f );
    part.min = -part.max;

    const auto [begin, end] = splitByChunks( totalSize, bufferSize, layerSize * layerOverlap );
    return cudaPipeline( part, begin, end,
        [&] ( MR::SimpleVolumeMinMax& part, Chunk chunk ) -> Expected<void>
        {
            pointsToDistanceVolumeKernel( cudaNodes.data(), cudaPoints.data(), cudaNormals.data(), cudaVolume.data(), cudaParams, chunk.size, chunk.offset );
            CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );

            CUDA_LOGE_RETURN_UNEXPECTED( cudaVolume.toVector( part.data.vec_ ) );

            return {};
        },
        [&] ( MR::SimpleVolumeMinMax& part, Chunk chunk )
        {
            part.dims.z = int( chunk.size / layerSize );
            return addPart( part, int( chunk.offset / layerSize ) );
        }
    );
}

} //namespace Cuda

} //namespace MR
#endif
