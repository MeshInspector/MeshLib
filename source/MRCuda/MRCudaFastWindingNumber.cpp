#include "MRCudaFastWindingNumber.h"
#include "MRCudaFastWindingNumber.cuh"
#include "MRCudaMath.cuh"
#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRDipole.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRTimer.h"

namespace MR
{
namespace Cuda
{

struct FastWindingNumberData
{
    DynamicArray<Dipole> dipoles;
    DynamicArray<float3> cudaPoints;
    DynamicArray<float3> cudaMeshPoints;
    DynamicArray<Node3> cudaNodes;
    DynamicArray<FaceToThreeVerts> cudaFaces;

    Matrix4 gridToMeshXf;
};

FastWindingNumber::FastWindingNumber( const Mesh& mesh ) : mesh_( mesh )
{
}

Expected<void> FastWindingNumber::prepareData_( ProgressCallback cb )
{
    CUDA_EXEC_RETURN_UNEXPECTED( cudaSetDevice( 0 ) );
    if ( data_ )
    {
        if ( !reportProgress( cb, 1.0f ) )
            return unexpectedOperationCanceled();
        return {};
    }
    MR_TIMER

    auto data = std::make_shared<FastWindingNumberData>();

    if ( !reportProgress( cb, 0.0f ) )
        return unexpectedOperationCanceled();

    CUDA_RETURN_UNEXPECTED( data->cudaMeshPoints.fromVector( mesh_.points.vec_ ) );
    if ( !reportProgress( cb, 0.1f ) )
        return unexpectedOperationCanceled();

    CUDA_RETURN_UNEXPECTED( data->cudaFaces.fromVector( mesh_.topology.getTriangulation().vec_ ) );
    if ( !reportProgress( cb, 0.3f ) )
        return unexpectedOperationCanceled();

    const AABBTree& tree = mesh_.getAABBTree();
    if ( !reportProgress( cb, 0.5f ) )
        return unexpectedOperationCanceled();

    const auto& nodes = tree.nodes();
    CUDA_RETURN_UNEXPECTED( data->cudaNodes.fromVector( nodes.vec_ ) );
    if ( !reportProgress( cb, 0.6f ) )
        return unexpectedOperationCanceled();

    CUDA_RETURN_UNEXPECTED( data->dipoles.fromVector( mesh_.getDipoles().vec_ ) );
    if ( !reportProgress( cb, 1.0f ) )
        return unexpectedOperationCanceled();

    data_ = std::move( data );
    return {};
}

void FastWindingNumber::calcFromVector( std::vector<float>& res, const std::vector<Vector3f>& points, float beta, FaceId skipFace )
{
    MR_TIMER
    prepareData_( {} ); //TODO: check error

    const size_t size = points.size();
    res.resize( size );
    data_->cudaPoints.fromVector( points );
    DynamicArrayF cudaResult( size );

    fastWindingNumberFromVector( data_->cudaPoints.data(), data_->dipoles.data(), data_->cudaNodes.data(), data_->cudaMeshPoints.data(), data_->cudaFaces.data(), cudaResult.data(), beta, int( skipFace ), size );
    CUDA_EXEC( cudaGetLastError() );

    CUDA_EXEC( cudaResult.toVector( res ) );
}

bool FastWindingNumber::calcSelfIntersections( FaceBitSet& res, float beta, ProgressCallback cb )
{
    MR_TIMER
    if ( !prepareData_( subprogress( cb, 0.0f, 0.5f ) ) )
        return false;

    const size_t size = mesh_.topology.faceSize();
    DynamicArrayF cudaResult( size );

    fastWindingNumberFromMesh(data_->dipoles.data(), data_->cudaNodes.data(), data_->cudaMeshPoints.data(), data_->cudaFaces.data(), cudaResult.data(), beta, size);
    if ( CUDA_EXEC( cudaGetLastError() ) )
        return false;

    std::vector<float> wns;
    if ( CUDA_EXEC( cudaResult.toVector( wns ) ) )
        return false;
    if ( !reportProgress( cb, 0.9f ) )
        return false;
    
    res.resize( size );
    return BitSetParallelForAll( res, [&] (FaceId f)
    {
        if ( wns[f] < 0 || wns[f] > 1 )
            res.set( f );
    }, subprogress( cb, 0.9f, 1.0f ) );
}

Expected<void> FastWindingNumber::calcFromGrid( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, float beta, ProgressCallback cb )
{
    MR_TIMER
    if ( auto maybe = prepareData_( {} ) )
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
    DynamicArrayF cudaResult( size );
    if ( !reportProgress( cb, 0.0f ) )
        return unexpectedOperationCanceled();

    fastWindingNumberFromGrid(
        int3{ dims.x, dims.y, dims.z },
        cudaGridToMeshXf,
        data_->dipoles.data(), data_->cudaNodes.data(), data_->cudaMeshPoints.data(), data_->cudaFaces.data(),
        cudaResult.data(), beta );
    
    CUDA_EXEC_RETURN_UNEXPECTED( cudaGetLastError() );

    CUDA_RETURN_UNEXPECTED( cudaResult.toVector( res ) );

    if ( !reportProgress( cb, 1.0f ) )
        return unexpectedOperationCanceled();
    return {};
}

Expected<void> FastWindingNumber::calcFromGridWithDistances( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, const DistanceToMeshOptions& options, const ProgressCallback& cb )
{
    MR_TIMER
    if ( auto maybe = prepareData_( {} ) )
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
    DynamicArrayF cudaResult( size );
    if ( !reportProgress( cb, 0.0f ) )
        return unexpectedOperationCanceled();

    signedDistance(
        int3{ dims.x, dims.y, dims.z },
        cudaGridToMeshXf,
        data_->dipoles.data(), data_->cudaNodes.data(), data_->cudaMeshPoints.data(), data_->cudaFaces.data(),
        cudaResult.data(), options );

    CUDA_EXEC_RETURN_UNEXPECTED( cudaGetLastError() );

    CUDA_RETURN_UNEXPECTED( cudaResult.toVector( res ) );

    if ( !reportProgress( cb, 1.0f ) )
        return unexpectedOperationCanceled();
    return {};
}

} //namespace Cuda

} //namespace MR
