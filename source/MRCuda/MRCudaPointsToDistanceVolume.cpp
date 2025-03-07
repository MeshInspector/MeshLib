#include "MRCudaPointsToDistanceVolume.h"
#ifndef MRCUDA_NO_VOXELS
#include "MRCudaPointsToDistanceVolume.cuh"

#include "MRCudaBasic.h"
#include "MRCudaMath.h"

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

MRCUDA_API Expected<void> pointsToDistanceVolumeByParts( const PointCloud& cloud,
    const MR::PointsToDistanceVolumeParams& params,
    std::function<void ( int )> setLayersPerBlock,
    std::function<Expected<void> ( const SimpleVolumeMinMax&, int )> addVolumePart )
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
        .origin = fromVec( params.origin ),
        .voxelSize = fromVec( params.voxelSize ),
        .dimensions = fromVec( params.dimensions ),
        .sigma = params.sigma,
        .minWeight = params.minWeight,
    };

    const auto layerSize = (size_t)params.dimensions.x * params.dimensions.y;
    const auto blockSize = maxBlockSize( getCudaSafeMemoryLimit(), params.dimensions, sizeof( float ) ).z;
    if ( setLayersPerBlock )
        setLayersPerBlock( blockSize );

    DynamicArrayF cudaVolume;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaVolume.resize( blockSize * layerSize ) );

    std::array<MR::SimpleVolumeMinMax, 2> volumes;
    for ( auto& vol : volumes )
    {
        vol.dims = params.dimensions;
        vol.voxelSize = params.voxelSize;
        vol.max = params.sigma * std::exp( -0.5f );
        vol.min = -vol.max;
    }

    int prevOffset = -1;
    for ( const auto [offset, size] : splitByChunks( params.dimensions.z, blockSize ) )
    {
        pointsToDistanceVolumeKernel( cudaNodes.data(), cudaPoints.data(), cudaNormals.data(), cudaVolume.data(), cudaParams, size * layerSize, offset * layerSize );

        if ( prevOffset >= 0 )
        {
            assert( !volumes[1].data.empty() );
            if ( auto res = addVolumePart( volumes[1], prevOffset ); !res )
                return unexpected( res.error() );
        }

        CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );

        volumes[0].dims.z = (int)size;
        CUDA_LOGE_RETURN_UNEXPECTED( cudaVolume.toVector( volumes[0].data ) );

        std::swap( volumes[0], volumes[1] );
        prevOffset = (int)offset;
    }
    // ...
    assert( !volumes[1].data.empty() );
    if ( auto res = addVolumePart( volumes[1], prevOffset ); !res )
        return unexpected( res.error() );

    return {};
}

} //namespace Cuda

} //namespace MR
#endif
