#include "MRCudaFastWindingNumber.h"
#include "MRCudaFastWindingNumber.cuh"

#include "MRCudaBasic.h"
#include "MRCudaMath.h"
#include "MRCudaMath.cuh"
#include "MRCudaPipeline.h"

#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRChunkIterator.h"
#include "MRMesh/MRDipole.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRTimer.h"
#include "MRPch/MRSpdlog.h"

#define RETURN_UNEXPECTED( expr ) if ( auto res = ( expr ); !res ) return MR::unexpected( std::move( res.error() ) )

namespace MR
{
namespace Cuda
{

struct FastWindingNumberDataBuffers
{
    DynamicArray<Dipole> dipoles;
    DynamicArray<float3> cudaMeshPoints;
    DynamicArray<Node3> cudaNodes;
    DynamicArray<FaceToThreeVerts> cudaFaces;

    [[nodiscard]] FastWindingNumberData toData() const
    {
        return {
            .dipoles = dipoles.data(),
            .nodes = cudaNodes.data(),
            .meshPoints = cudaMeshPoints.data(),
            .faces = cudaFaces.data(),
        };
    }
};

FastWindingNumber::FastWindingNumber( const Mesh& mesh ) : mesh_( mesh )
{
}

Expected<void> FastWindingNumber::prepareData_( ProgressCallback cb )
{
    CUDA_LOGE_RETURN_UNEXPECTED( cudaSetDevice( 0 ) );
    if ( data_ )
    {
        if ( !reportProgress( cb, 1.0f ) )
            return unexpectedOperationCanceled();
        return {};
    }
    MR_TIMER;

    auto data = std::make_shared<FastWindingNumberDataBuffers>();

    if ( !reportProgress( cb, 0.0f ) )
        return unexpectedOperationCanceled();

    CUDA_LOGE_RETURN_UNEXPECTED( data->cudaMeshPoints.fromVector( mesh_.points.vec_ ) );
    if ( !reportProgress( cb, 0.1f ) )
        return unexpectedOperationCanceled();

    CUDA_LOGE_RETURN_UNEXPECTED( data->cudaFaces.fromVector( mesh_.topology.getTriangulation().vec_ ) );
    if ( !reportProgress( cb, 0.3f ) )
        return unexpectedOperationCanceled();

    const AABBTree& tree = mesh_.getAABBTree();
    if ( !reportProgress( cb, 0.5f ) )
        return unexpectedOperationCanceled();

    const auto& nodes = tree.nodes();
    CUDA_LOGE_RETURN_UNEXPECTED( data->cudaNodes.fromVector( nodes.vec_ ) );
    if ( !reportProgress( cb, 0.6f ) )
        return unexpectedOperationCanceled();

    CUDA_LOGE_RETURN_UNEXPECTED( data->dipoles.fromVector( mesh_.getDipoles().vec_ ) );
    if ( !reportProgress( cb, 1.0f ) )
        return unexpectedOperationCanceled();

    data_ = std::move( data );
    return {};
}

Expected<void> FastWindingNumber::calcFromVector( std::vector<float>& res, const std::vector<Vector3f>& points, float beta, FaceId skipFace, const ProgressCallback& cb )
{
    MR_TIMER;
    return prepareData_( subprogress( cb, 0.0, 0.5f ) ).and_then( [&]() -> Expected<void>
    {
        const auto totalSize = points.size();
        const auto bufferSize = maxBufferSize( getCudaSafeMemoryLimit(), totalSize, sizeof( float ) + sizeof( float3 ) );

        DynamicArray<float3> cudaPoints;
        CUDA_LOGE_RETURN_UNEXPECTED( cudaPoints.resize( bufferSize ) );
        if ( !reportProgress( cb, 0.6f ) )
            return unexpectedOperationCanceled();

        DynamicArrayF cudaResult;
        CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.resize( bufferSize ) );

        res.resize( points.size() );

        const auto cb1 = subprogress( cb, 0.60f, 1.00f );
        const auto iterCount = chunkCount( totalSize, bufferSize );
        size_t iterIndex = 0;

        for ( const auto [offset, size] : splitByChunks( totalSize, bufferSize ) )
        {
            const auto cb2 = subprogress( cb1, iterIndex++, iterCount );

            CUDA_LOGE_RETURN_UNEXPECTED( cudaPoints.copyFrom( points.data() + offset, size ) );

            fastWindingNumberFromVector( cudaPoints.data(), data_->toData(), cudaResult.data(), beta, int( skipFace ), size );
            CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );
            if ( !reportProgress( cb2, 0.25f ) )
                return unexpectedOperationCanceled();

            CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.copyTo( res.data() + offset, size ) );
            if ( !reportProgress( cb2, 1.00f ) )
                return unexpectedOperationCanceled();
        }

        return {};
    } );
}

Expected<void> FastWindingNumber::calcSelfIntersections( FaceBitSet& res, float beta, const ProgressCallback& cb )
{
    MR_TIMER;
    return prepareData_( subprogress( cb, 0.0, 0.5f ) ).and_then( [&]() -> Expected<void>
    {
        const auto totalSize = mesh_.topology.faceSize();
        const auto bufferSize = maxBufferSize( getCudaSafeMemoryLimit(), totalSize, sizeof( float ) );

        DynamicArrayF cudaResult;
        CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.resize( bufferSize ) );
        if ( !reportProgress( cb, 0.6f ) )
            return unexpectedOperationCanceled();

        std::vector<float> wns;
        wns.resize( totalSize );

        const auto cb1 = subprogress( cb, 0.60f, 0.90f );
        const auto iterCount = chunkCount( totalSize, bufferSize );
        size_t iterIndex = 0;

        for ( const auto [offset, size] : splitByChunks( totalSize, bufferSize ) )
        {
            const auto cb2 = subprogress( cb1, iterIndex++, iterCount );

            fastWindingNumberFromMesh( data_->toData(), cudaResult.data(), beta, size, offset );
            CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );
            if ( !reportProgress( cb2, 0.33f ) )
                return unexpectedOperationCanceled();

            CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.copyTo( wns.data() + offset, size ) );
            if ( !reportProgress( cb2, 1.00f ) )
                return unexpectedOperationCanceled();
        }

        res.resize( totalSize );
        if ( !BitSetParallelForAll( res, [&] (FaceId f)
        {
            if ( wns[f] < 0 || wns[f] > 1 )
                res.set( f );
        }, subprogress( cb, 0.9f, 1.0f ) ) )
            return unexpectedOperationCanceled();

        return {};
    } );
}

Expected<void> FastWindingNumber::calcFromGrid( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, float beta, const ProgressCallback& cb )
{
    MR_TIMER;
    if ( auto maybe = prepareData_( subprogress( cb, 0.0, 0.5f ) ); !maybe )
        return unexpected( std::move( maybe.error() ) );

    const auto getCudaMatrix = [] ( const AffineXf3f& xf )
    {
        Matrix4 res;
        res.x.x = xf.A.x.x; res.x.y = xf.A.x.y; res.x.z = xf.A.x.z;
        res.y.x = xf.A.y.x; res.y.y = xf.A.y.y; res.y.z = xf.A.y.z;
        res.z.x = xf.A.z.x; res.z.y = xf.A.z.y; res.z.z = xf.A.z.z;
        res.b.x = xf.b.x; res.b.y = xf.b.y; res.b.z = xf.b.z;
        res.isIdentity = false;
        return res;
    };
    const Matrix4 cudaGridToMeshXf = ( gridToMeshXf == AffineXf3f{} ) ? Matrix4{} : getCudaMatrix( gridToMeshXf );

    const auto totalSize = (size_t)dims.x * dims.y * dims.z;
    const auto bufferSize = maxBufferSizeAlignedByBlock( getCudaSafeMemoryLimit(), dims, sizeof( float ) );

    DynamicArrayF cudaResult;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.resize( bufferSize ) );
    if ( !reportProgress( cb, 0.6f ) )
        return unexpectedOperationCanceled();

    const auto cb1 = subprogress( cb, 0.60f, 1.00f );
    const auto iterCount = chunkCount( totalSize, bufferSize );
    size_t iterIndex = 0;

    for ( const auto [offset, size] : splitByChunks( totalSize, bufferSize ) )
    {
        const auto cb2 = subprogress( cb1, iterIndex++, iterCount );

        fastWindingNumberFromGrid(
            int3 { dims.x, dims.y, dims.z },
            cudaGridToMeshXf,
            data_->toData(),
            cudaResult.data(),
            beta,
            size,
            offset
        );
        CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );
        if ( !reportProgress( cb2, 0.25f ) )
            return unexpectedOperationCanceled();

        CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.copyTo( res.data() + offset, size ) );
        if ( !reportProgress( cb2, 1.00f ) )
            return unexpectedOperationCanceled();
    }

    return {};
}

Expected<void> FastWindingNumber::calcFromGridWithDistances( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, const DistanceToMeshOptions& options, const ProgressCallback& cb )
{
    MR_TIMER;
    if ( auto maybe = prepareData_( subprogress( cb, 0.0, 0.5f ) ); !maybe )
        return unexpected( std::move( maybe.error() ) );

    const auto getCudaMatrix = [] ( const AffineXf3f& xf )
    {
        Matrix4 res;
        res.x.x = xf.A.x.x; res.x.y = xf.A.x.y; res.x.z = xf.A.x.z;
        res.y.x = xf.A.y.x; res.y.y = xf.A.y.y; res.y.z = xf.A.y.z;
        res.z.x = xf.A.z.x; res.z.y = xf.A.z.y; res.z.z = xf.A.z.z;
        res.b.x = xf.b.x; res.b.y = xf.b.y; res.b.z = xf.b.z;
        res.isIdentity = false;
        return res;
    };
    const Matrix4 cudaGridToMeshXf = ( gridToMeshXf == AffineXf3f{} ) ? Matrix4{} : getCudaMatrix( gridToMeshXf );

    const auto totalSize = (size_t)dims.x * dims.y * dims.z;
    const auto bufferSize = maxBufferSizeAlignedByBlock( getCudaSafeMemoryLimit(), dims, sizeof( float ) );

    DynamicArrayF cudaResult;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.resize( bufferSize ) );
    if ( !reportProgress( cb, 0.6f ) )
        return unexpectedOperationCanceled();

    const auto cb1 = subprogress( cb, 0.60f, 1.00f );
    const auto iterCount = chunkCount( totalSize, bufferSize );
    size_t iterIndex = 0;

    for ( const auto [offset, size] : splitByChunks( totalSize, bufferSize ) )
    {
        const auto cb2 = subprogress( cb1, iterIndex++, iterCount );

        signedDistance(
            int3 { dims.x, dims.y, dims.z },
            cudaGridToMeshXf,
            data_->toData(),
            cudaResult.data(),
            size,
            offset,
            options
        );
        CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );
        if ( !reportProgress( cb2, 0.25f ) )
            return unexpectedOperationCanceled();

        CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.copyTo( res.data() + offset, size ) );
        if ( !reportProgress( cb2, 1.00f ) )
            return unexpectedOperationCanceled();
    }

    return {};
}

Expected<void> FastWindingNumber::calcFromGridByParts( GridByPartsFunc resFunc, const Vector3i& dims, const AffineXf3f& gridToMeshXf, float beta, int layerOverlap, const ProgressCallback& cb )
{
    MR_TIMER;

    if ( auto maybe = prepareData_( subprogress( cb, 0.0, 0.5f ) ); !maybe )
        return unexpected( std::move( maybe.error() ) );

    const auto cudaGridToMeshXf = fromXf( gridToMeshXf );

    const auto layerSize = (size_t)dims.x * dims.y;
    const auto totalSize = layerSize * dims.z;
    const auto bufferSize = maxBufferSizeAlignedByBlock( getCudaSafeMemoryLimit(), dims, sizeof( float ) );
    // not spdlog::debug to see it in Release logs
    spdlog::info(
        "CudaFastWindingNumber: Required memory: {}, available memory: {}, iterations: {}",
        bytesString( totalSize * sizeof( float ) ),
        bytesString( bufferSize * sizeof( float ) ),
        chunkCount( totalSize, bufferSize )
    );

    DynamicArrayF cudaResult;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.resize( bufferSize ) );
    if ( !reportProgress( cb, 0.6f ) )
        return unexpectedOperationCanceled();

    const auto cb1 = subprogress( cb, 0.60f, 1.00f );
    const auto iterCount = chunkCount( totalSize, bufferSize );
    size_t iterIndex = 0;

    const auto [begin, end] = splitByChunks( totalSize, bufferSize, layerSize * layerOverlap );
    return cudaPipeline( std::vector<float>{}, begin, end,
        [&] ( std::vector<float>& data, Chunk chunk ) -> Expected<void>
        {
            spdlog::info( "CudaFastWindingNumber: chunk [{}, {}) starting", chunk.offset, chunk.offset + chunk.size );
            fastWindingNumberFromGrid(
                int3 { dims.x, dims.y, dims.z },
                cudaGridToMeshXf,
                data_->toData(),
                cudaResult.data(),
                beta,
                chunk.size,
                chunk.offset
            );
            CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );

            CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.toVector( data ) );
            spdlog::info( "CudaFastWindingNumber: chunk [{}, {}) computed", chunk.offset, chunk.offset + chunk.size );

            // it has to be checked only after the initial chuck (iterIndex == 0) when CPU does nothing,
            // for other chunks cancellation will be checked after CPU part
            if ( iterIndex == 0 && !reportProgress( subprogress( cb1, 0, iterCount ), 0.5f ) )
                return unexpectedOperationCanceled();

            return {};
        },
        [&] ( std::vector<float>& data, Chunk chunk ) -> Expected<void>
        {
            const auto cb2 = subprogress( cb1, iterIndex++, iterCount );
            data.resize( chunk.size );
            RETURN_UNEXPECTED( resFunc(
                std::move( data ),
                { dims.x, dims.y, int( chunk.size / layerSize ) },
                int( chunk.offset / layerSize )
            ) );
            if ( !reportProgress( cb2, 1.00f ) )
                return unexpectedOperationCanceled();
            // make sure the vector is valid
            data.clear();
            return {};
        }
    );
}

Expected<void> FastWindingNumber::calcFromGridWithDistancesByParts( GridByPartsFunc resFunc, const Vector3i& dims, const AffineXf3f& gridToMeshXf, const DistanceToMeshOptions& options, int layerOverlap, const ProgressCallback& cb )
{
    MR_TIMER;

    if ( auto maybe = prepareData_( subprogress( cb, 0.0, 0.5f ) ); !maybe )
        return unexpected( std::move( maybe.error() ) );

    const auto cudaGridToMeshXf = fromXf( gridToMeshXf );

    const auto layerSize = (size_t)dims.x * dims.y;
    const auto totalSize = layerSize * dims.z;
    const auto bufferSize = maxBufferSizeAlignedByBlock( getCudaSafeMemoryLimit(), dims, sizeof( float ) );
    // not spdlog::debug to see it in Release logs
    spdlog::info(
        "CudaFastWindingNumber+dist: Required memory: {}, available memory: {}, iterations: {}",
        bytesString( totalSize * sizeof( float ) ),
        bytesString( bufferSize * sizeof( float ) ),
        chunkCount( totalSize, bufferSize )
    );

    DynamicArrayF cudaResult;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.resize( bufferSize ) );
    if ( !reportProgress( cb, 0.6f ) )
        return unexpectedOperationCanceled();

    const auto cb1 = subprogress( cb, 0.60f, 1.00f );
    const auto iterCount = chunkCount( totalSize, bufferSize );
    size_t iterIndex = 0;

    const auto [begin, end] = splitByChunks( totalSize, bufferSize, layerSize * layerOverlap );
    return cudaPipeline( std::vector<float>{}, begin, end,
        [&] ( std::vector<float>& data, Chunk chunk ) -> Expected<void>
        {
            spdlog::info( "CudaFastWindingNumber+dist: chunk [{}, {}) starting", chunk.offset, chunk.offset + chunk.size );
            signedDistance(
                int3 { dims.x, dims.y, dims.z },
                cudaGridToMeshXf,
                data_->toData(),
                cudaResult.data(),
                chunk.size,
                chunk.offset,
                options
            );
            CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );

            CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.toVector( data ) );
            spdlog::info( "CudaFastWindingNumber+dist: chunk [{}, {}) computed", chunk.offset, chunk.offset + chunk.size );

            // it has to be checked only after the initial chuck (iterIndex == 0) when CPU does nothing,
            // for other chunks cancellation will be checked after CPU part
            if ( iterIndex == 0 && !reportProgress( subprogress( cb1, 0, iterCount ), 0.5f ) )
                return unexpectedOperationCanceled();

            return {};
        },
        [&] ( std::vector<float>& data, Chunk chunk ) -> Expected<void>
        {
            const auto cb2 = subprogress( cb1, iterIndex++, iterCount );
            data.resize( chunk.size );
            RETURN_UNEXPECTED( resFunc(
                std::move( data ),
                { dims.x, dims.y, int( chunk.size / layerSize ) },
                int( chunk.offset / layerSize )
            ) );
            if ( !reportProgress( cb2, 1.00f ) )
                return unexpectedOperationCanceled();
            // make sure the vector is valid
            data.clear();
            return {};
        }
    );
}

} //namespace Cuda

} //namespace MR
