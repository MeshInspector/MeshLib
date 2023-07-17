#include "MRCudaFastWindingNumber.h"
#include "MRCudaFastWindingNumber.cuh"
#include "MRCudaMath.cuh"
#include "MRMesh/MRAABBTree.h"
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
    DynamicArrayF cudaResult;
    DynamicArray<float3> cudaMeshPoints;
    DynamicArray<Node3> cudaNodes;
    DynamicArray<FaceToThreeVerts> cudaFaces;

    Matrix4 gridToMeshXf;
};

FastWindingNumber::FastWindingNumber( const Mesh& mesh ) : mesh_( mesh )
{
}

bool FastWindingNumber::prepareData_( ProgressCallback cb )
{
    CUDA_EXEC( cudaSetDevice( 0 ) );
    if ( data_ )
        return reportProgress( cb, 1.0f );
    MR_TIMER

    auto data = std::make_shared<FastWindingNumberData>();

    if ( !reportProgress( cb, 0.0f ) )
        return false;
    const AABBTree& tree = mesh_.getAABBTree();
    Dipoles dipoles;
    calcDipoles( dipoles, tree, mesh_ );
    if ( !reportProgress( cb, 0.5f ) )
        return false;

    data->dipoles.fromVector( dipoles.vec_ );
    if ( !reportProgress( cb, 0.625f ) )
        return false;

    const auto& nodes = tree.nodes();
    const auto& meshPoints = mesh_.points;
    const auto tris = mesh_.topology.getTriangulation();

    data->cudaMeshPoints.fromVector( meshPoints.vec_ );
    if ( !reportProgress( cb, 0.75f ) )
        return false;

    data->cudaNodes.fromVector( nodes.vec_ );
    if ( !reportProgress( cb, 0.875f ) )
        return false;

    data->cudaFaces.fromVector( tris.vec_ );
    if ( !reportProgress( cb, 1.0f ) )
        return false;

    data_ = std::move( data );
    return true;
}

void FastWindingNumber::calcFromVector( std::vector<float>& res, const std::vector<Vector3f>& points, float beta, FaceId skipFace )
{
    MR_TIMER
    prepareData_( {} );

    const size_t size = points.size();
    res.resize( size );
    data_->cudaPoints.fromVector( points );
    data_->cudaResult.resize( size );

    fastWindingNumberFromVector( data_->cudaPoints.data(), data_->dipoles.data(), data_->cudaNodes.data(), data_->cudaMeshPoints.data(), data_->cudaFaces.data(), data_->cudaResult.data(), beta, int( skipFace ), size );
    CUDA_EXEC( cudaGetLastError() );

    data_->cudaResult.toVector( res );
}

bool FastWindingNumber::calcSelfIntersections( FaceBitSet& res, float beta, ProgressCallback cb )
{
    MR_TIMER
    if ( !prepareData_( subprogress( cb, 0.0f, 0.5f ) ) )
        return false;

    const size_t size = mesh_.topology.faceSize();
    res.resize( size );
    data_->cudaResult.resize( size );

    fastWindingNumberFromMesh(data_->dipoles.data(), data_->cudaNodes.data(), data_->cudaMeshPoints.data(), data_->cudaFaces.data(), data_->cudaResult.data(), beta, size);
    if ( CUDA_EXEC( cudaGetLastError() ) )
        return false;

    std::vector<float> wns;
    data_->cudaResult.toVector( wns );
    if ( !reportProgress( cb, 0.9f ) )
        return false;
    
    return BitSetParallelForAll( res, [&] (FaceId f)
    {
        if ( wns[f] < 0 || wns[f] > 1 )
            res.set( f );
    }, subprogress( cb, 0.9f, 1.0f ) );
}

VoidOrErrStr FastWindingNumber::calcFromGrid( std::vector<float>& res, const Vector3i& dims, const Vector3f& minCoord, const Vector3f& voxelSize, const AffineXf3f& gridToMeshXf, float beta, ProgressCallback cb )
{
    MR_TIMER
    prepareData_( {} );

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
    data_->cudaResult.resize( size );
    if ( !reportProgress( cb, 0.0f ) )
        return unexpectedOperationCanceled();

    fastWindingNumberFromGrid(
        int3{ dims.x, dims.y, dims.z },
        float3{ minCoord.x, minCoord.y, minCoord.z },
        float3{ voxelSize.x, voxelSize.y, voxelSize.z }, cudaGridToMeshXf,
        data_->dipoles.data(), data_->cudaNodes.data(), data_->cudaMeshPoints.data(), data_->cudaFaces.data(),
        data_->cudaResult.data(), beta );
    
    if ( auto code = CUDA_EXEC( cudaGetLastError() ) )
        return unexpected( Cuda::getError( code ) );

    if ( auto code = data_->cudaResult.toVector( res ) )
        return unexpected( Cuda::getError( code ) );

    if ( !reportProgress( cb, 1.0f ) )
        return unexpectedOperationCanceled();
    return {};
}

VoidOrErrStr FastWindingNumber::calcFromGridWithDistances( std::vector<float>& res, const Vector3i& dims, const Vector3f& minCoord, const Vector3f& voxelSize, const AffineXf3f& gridToMeshXf, float beta, float maxDistSq, float minDistSq, ProgressCallback cb )
{
    MR_TIMER
    prepareData_( {} );

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
    data_->cudaResult.resize( size );
    if ( !reportProgress( cb, 0.0f ) )
        return unexpectedOperationCanceled();

    signedDistance(
        int3{ dims.x, dims.y, dims.z },
        float3{ minCoord.x, minCoord.y, minCoord.z },
        float3{ voxelSize.x, voxelSize.y, voxelSize.z }, cudaGridToMeshXf,
        data_->dipoles.data(), data_->cudaNodes.data(), data_->cudaMeshPoints.data(), data_->cudaFaces.data(),
        data_->cudaResult.data(), beta, maxDistSq, minDistSq );

    if ( auto code = CUDA_EXEC( cudaGetLastError() ) )
        return unexpected( Cuda::getError( code ) );

    if ( auto code = data_->cudaResult.toVector( res ) )
        return unexpected( Cuda::getError( code ) );

    if ( !reportProgress( cb, 1.0f ) )
        return unexpectedOperationCanceled();
    return {};

}

size_t FastWindingNumber::fromVectorHeapBytes( size_t inputSize ) const
{
    size_t currentSize = 0;
    if ( data_ )
    {
        currentSize += data_->cudaPoints.size() * sizeof( float3 );
        currentSize += data_->cudaResult.size() * sizeof( float );
    }
    size_t newSize = inputSize * ( sizeof( float3 ) + sizeof( float ) );
    if ( newSize <= currentSize )
        return 0;
    return newSize - currentSize;
}

size_t FastWindingNumber::selfIntersectionsHeapBytes( const Mesh& mesh ) const
{
    size_t currentSize = 0;
    if ( data_ )
        currentSize += data_->cudaResult.size() * sizeof( float );
    size_t newSize = mesh.topology.faceSize() * sizeof( float );
    if ( newSize <= currentSize )
        return 0;
    return newSize - currentSize;
}

size_t FastWindingNumber::fromGridHeapBytes( const Vector3i& dims ) const
{
    size_t currentSize = 0;
    if ( data_ )
        currentSize += data_->cudaResult.size() * sizeof( float );
    size_t newSize = size_t( dims.x ) * dims.y * dims.z * sizeof( float );
    if ( newSize <= currentSize )
        return 0;
    return newSize - currentSize;
}

} //namespace Cuda

} //namespace MR
