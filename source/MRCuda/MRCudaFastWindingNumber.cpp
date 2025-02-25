#include "MRCudaFastWindingNumber.h"
#include "MRCudaFastWindingNumber.cuh"

#include "MRCudaBasic.h"
#include "MRCudaMath.cuh"

#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRChunkIterator.h"
#include "MRMesh/MRDipole.h"
#include "MRMesh/MRTimer.h"

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
    MR_TIMER

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
    MR_TIMER
    return prepareData_( subprogress( cb, 0.0, 0.5f ) ).and_then( [&]() -> Expected<void>
    {
        const size_t size = points.size();
        res.resize( size );

        DynamicArray<float3> cudaPoints;
        CUDA_LOGE_RETURN_UNEXPECTED( cudaPoints.fromVector( points ) );
        if ( !reportProgress( cb, 0.6f ) )
            return unexpectedOperationCanceled();

        DynamicArrayF cudaResult;
        CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.resize( size ) );
        fastWindingNumberFromVector( cudaPoints.data(), data_->toData(), cudaResult.data(), beta, int( skipFace ), size );
        CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );
        if ( !reportProgress( cb, 0.7f ) )
            return unexpectedOperationCanceled();

        CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.toVector( res ) );
        if ( !reportProgress( cb, 1.0f ) )
            return unexpectedOperationCanceled();
        return {};
    } );
}

Expected<void> FastWindingNumber::calcSelfIntersections( FaceBitSet& res, float beta, const ProgressCallback& cb )
{
    MR_TIMER
    return prepareData_( subprogress( cb, 0.0, 0.5f ) ).and_then( [&]() -> Expected<void>
    {
        const size_t size = mesh_.topology.faceSize();
        DynamicArrayF cudaResult;
        CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.resize( size ) );
        if ( !reportProgress( cb, 0.6f ) )
            return unexpectedOperationCanceled();

        fastWindingNumberFromMesh( data_->toData(), cudaResult.data(), beta, size);
        CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );
        if ( !reportProgress( cb, 0.7f ) )
            return unexpectedOperationCanceled();

        std::vector<float> wns;
        CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.toVector( wns ) );
        if ( !reportProgress( cb, 0.9f ) )
            return unexpectedOperationCanceled();
    
        res.resize( size );
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
    MR_TIMER
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
    const size_t size = size_t( dims.x ) * dims.y * dims.z;
    DynamicArrayF cudaResult;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.resize( size ) );
    if ( !reportProgress( cb, 0.6f ) )
        return unexpectedOperationCanceled();

    fastWindingNumberFromGrid(
        int3{ dims.x, dims.y, dims.z },
        cudaGridToMeshXf,
        data_->toData(),
        cudaResult.data(), beta );
    CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );
    if ( !reportProgress( cb, 0.7f ) )
        return unexpectedOperationCanceled();

    CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.toVector( res ) );
    if ( !reportProgress( cb, 1.0f ) )
        return unexpectedOperationCanceled();
    return {};
}

Expected<void> FastWindingNumber::calcFromGridWithDistances( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, const DistanceToMeshOptions& options, const ProgressCallback& cb )
{
    MR_TIMER
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

    // TODO: allow user to set the upper limit
    const auto maxBufferBytes = getCudaAvailableMemory();
    const auto maxBufferSize = maxBufferBytes / sizeof( float );

    const auto layerSize = size_t( dims.x ) * dims.y;
    const auto maxLayerCountInBuffer = maxBufferSize / layerSize;
    const auto totalSize = dims.z * layerSize;
    const auto bufferSize = std::min( maxLayerCountInBuffer * layerSize, totalSize );

    DynamicArrayF cudaResult;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.resize( bufferSize ) );
    if ( !reportProgress( cb, 0.6f ) )
        return unexpectedOperationCanceled();

    const auto cb1 = subprogress( cb, 0.60f, 1.00f );
    const auto iterCount = chunkCount( totalSize, bufferSize );
    for ( const auto chunk : splitByChunks( totalSize, bufferSize ) )
    {
        const auto cb2 = subprogress( cb1, chunk.index, iterCount );

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
        if ( !reportProgress( cb2, 0.25f ) )
            return unexpectedOperationCanceled();

        CUDA_LOGE_RETURN_UNEXPECTED( cudaResult.copyTo( res.data() + chunk.offset, chunk.size ) );
        if ( !reportProgress( cb2, 1.00f ) )
            return unexpectedOperationCanceled();
    }

    return {};
}

} //namespace Cuda

} //namespace MR
