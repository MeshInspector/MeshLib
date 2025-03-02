#include "MRCudaPointsProject.cuh"
#include "MRCudaPointsProject.h"

#include "MRCudaBasic.cuh"
#include "MRCudaBasic.h"
#include "MRCudaMath.h"
#include "MRCudaPointCloud.h"

#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRChunkIterator.h"

static_assert( sizeof( MR::Cuda::PointsProjectionResult ) == sizeof( MR::PointsProjectionResult ) );

namespace MR::Cuda
{

Expected<std::vector<MR::PointsProjectionResult>> findProjectionOnPoints( const PointCloud& pointCloud,
    const std::vector<Vector3f>& points, const AffineXf3f* pointsXf, const AffineXf3f* refXf, float upDistLimitSq,
    float loDistLimitSq )
{
    auto cudaPointCloud = copyDataFrom( pointCloud );
    if ( !cudaPointCloud )
        return unexpected( cudaPointCloud.error() );

    const auto totalSize = points.size();
    const auto bufferSize = maxBufferSize( getCudaSafeMemoryLimit(), totalSize, sizeof( float3 ) + sizeof( PointsProjectionResult ) );

    DynamicArray<float3> cudaPoints;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaPoints.resize( bufferSize ) );

    DynamicArray<PointsProjectionResult> cudaResult;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.resize( bufferSize ) );

    std::vector<MR::PointsProjectionResult> results;
    results.resize( totalSize );

    const auto cudaPointsXf = pointsXf ? fromXf( *pointsXf ) : Matrix4{};
    const auto cudaRefXf = refXf ? fromXf( *refXf ) : Matrix4{};

    for ( const auto [offset, size] : splitByChunks( totalSize, bufferSize ) )
    {
        CUDA_LOGE_RETURN_UNEXPECTED( cudaPoints.copyFrom( points.data() + offset, size ) );

        findProjectionOnPointsKernel( cudaResult.data(), (*cudaPointCloud)->data(), cudaPoints.data(), cudaPointsXf, cudaRefXf, upDistLimitSq, loDistLimitSq, size, offset );
        CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );

        CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.copyTo( results.data() + offset, size ) );
    }

    return results;
}

size_t findProjectionOnPointsHeapBytes( const PointCloud& pointCloud, size_t pointsCount )
{
    constexpr size_t cMinCudaBufferSize = 1 << 24; // 16 MiB
    return
          pointCloudHeapBytes( pointCloud )
        + std::min( ( sizeof( float3 ) + sizeof( PointsProjectionResult ) ) * pointsCount, cMinCudaBufferSize );
}

} // namespace MR::Cuda
