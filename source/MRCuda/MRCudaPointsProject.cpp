#include "MRCudaPointsProject.cuh"
#include "MRCudaPointsProject.h"

#include "MRCudaBasic.cuh"
#include "MRCudaBasic.h"
#include "MRCudaMath.h"
#include "MRCudaPointCloud.h"

#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRChunkIterator.h"

static_assert( sizeof( MR::Cuda::PointsProjectionResult ) == sizeof( MR::PointsProjectionResult ) );

namespace MR::Cuda
{

Expected<std::vector<MR::PointsProjectionResult>> findProjectionOnPoints( const PointCloud& pointCloud,
    const std::vector<Vector3f>& points, const FindProjectionOnPointsSettings& settings )
{
    std::vector<MR::PointsProjectionResult> results;
    PointsProjector projector;
    return projector.setPointCloud( pointCloud )
        .and_then( [&] { return projector.findProjections( results, points, settings ); } )
        .transform( [&] { return results; } );
}

Expected<void> PointsProjector::setPointCloud( const PointCloud& pointCloud )
{
    if ( auto res = copyDataFrom( pointCloud ) )
    {
        data_ = std::move( *res );
        return {};
    }
    else
    {
        return unexpected( std::move( res.error() ) );
    }
}

Expected<void> PointsProjector::findProjections( std::vector<MR::PointsProjectionResult>& results,
    const std::vector<Vector3f>& points, const FindProjectionOnPointsSettings& settings ) const
{
    if ( !data_ )
        return unexpected( "No reference point cloud is set" );

    const auto totalSize = points.size();
    const auto bufferSize = maxBufferSize( getCudaSafeMemoryLimit(), totalSize, sizeof( float3 ) + sizeof( PointsProjectionResult ) );

    DynamicArray<float3> cudaPoints;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaPoints.resize( bufferSize ) );

    DynamicArray<PointsProjectionResult> cudaResult;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.resize( bufferSize ) );

    results.resize( totalSize );

    DynamicArray<uint64_t> cudaValid;
    if ( settings.valid )
    {
        assert( points.size() <= settings.valid->size() );
        std::vector<uint64_t> validVec;
        boost::to_block_range( *settings.valid, std::back_inserter( validVec ) );
        CUDA_LOGE_RETURN_UNEXPECTED( cudaValid.fromVector( validVec ) );
    }

    const auto cudaXf = settings.xf ? fromXf( *settings.xf ) : Matrix4{};

    for ( const auto [offset, size] : splitByChunks( totalSize, bufferSize ) )
    {
        CUDA_LOGE_RETURN_UNEXPECTED( cudaPoints.copyFrom( points.data() + offset, size ) );

        findProjectionOnPointsKernel( cudaResult.data(), data_->data(), cudaPoints.data(), settings.valid ? cudaValid.data() : nullptr, cudaXf, settings.upDistLimitSq, settings.loDistLimitSq, settings.skipSameIndex, size, offset );
        CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );

        CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.copyTo( results.data() + offset, size ) );
    }

    return {};
}

size_t findProjectionOnPointsHeapBytes( const PointCloud& pointCloud, size_t pointsCount )
{
    constexpr size_t cMinCudaBufferSize = 1 << 24; // 16 MiB
    return
          pointCloudHeapBytes( pointCloud )
        + std::min( ( sizeof( float3 ) + sizeof( PointsProjectionResult ) ) * pointsCount, cMinCudaBufferSize );
}

} // namespace MR::Cuda
