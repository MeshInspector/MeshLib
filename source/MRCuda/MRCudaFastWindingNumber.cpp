#include "MRCudaFastWindingNumber.h"
#include "MRCudaFastWindingNumber.cuh"
#include "MRCudaMesh.cuh"
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

FastWindingNumber::FastWindingNumber( const Mesh& mesh )
: IFastWindingNumber( mesh )
{
    data_ = std::make_shared<FastWindingNumberData>();

    const AABBTree& tree = mesh.getAABBTree();
    IFastWindingNumber::Dipoles dipoles;
    IFastWindingNumber::calcDipoles( dipoles, tree, mesh );
    data_->dipoles.fromVector( dipoles.vec_ );

    const auto& nodes = tree.nodes();
    const auto& meshPoints = mesh.points;
    const auto tris = mesh.topology.getTriangulation();

    data_->cudaMeshPoints.fromVector( meshPoints.vec_ );
    data_->cudaNodes.fromVector( nodes.vec_ );
    data_->cudaFaces.fromVector( tris.vec_ );
}

void FastWindingNumber::calcFromVector( std::vector<float>& res, const std::vector<Vector3f>& points, float beta, FaceId skipFace )
{
    cudaSetDevice( 0 );
    const size_t size = points.size();
    res.resize( size );
    data_->cudaPoints.fromVector( points );
    data_->cudaResult.resize( size );

    fastWindingNumberFromVectorKernel( data_->cudaPoints.data(), data_->dipoles.data(), data_->cudaNodes.data(), data_->cudaMeshPoints.data(), data_->cudaFaces.data(), data_->cudaResult.data(), beta, int( skipFace ), size );
    data_->cudaResult.toVector( res );
}

float triangleSolidAngle1( const float3& p, const float3& tri0, const float3& tri1, const float3& tri2 )
{
    const auto mx = tri0 - p;
    const auto my = tri1 - p;
    const auto mz = tri2 - p;

    const auto x = length( mx );
    const auto y = length( my );
    const auto z = length( mz );

    auto den = x * y * z + dot( mx, my ) * z + dot( my, mz ) * x + dot( mz, mx ) * y;
    return 2 * std::atan2( mx.x * ( my.y * mz.z - my.z * mz.y ) - mx.y * ( my.x * mz.z - my.z * mz.x ) + mx.z * ( my.x * mz.y - my.y * mz.x ), den );
}

void FastWindingNumber::calcSelfIntersections( FaceBitSet& res, float beta )
{
    cudaSetDevice( 0 );
    const size_t size = mesh_.topology.faceSize();
    res.resize( size );
    data_->cudaResult.resize( size );

    fastWindingNumberFromMeshKernel(data_->dipoles.data(), data_->cudaNodes.data(), data_->cudaMeshPoints.data(), data_->cudaFaces.data(), data_->cudaResult.data(), beta, size);
    std::vector<float> wns;
    data_->cudaResult.toVector( wns );
    
    BitSetParallelFor(res, [&] (FaceId f)
    {
        if ( wns[f] < 0 || wns[f] > 1 )
            res.set( f );
    } );   
}

void FastWindingNumber::calcFromGrid( std::vector<float>& res, const Vector3i& dims, const Vector3f& minCoord, const Vector3f& voxelSize, const AffineXf3f& gridToMeshXf, float beta )
{
    MR_TIMER

    cudaSetDevice( 0 );

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
    const size_t size = dims.x * dims.y * dims.z;
    data_->cudaResult.resize( size );    
    fastWindingNumberFromGridKernel( int3{ dims.x, dims.y, dims.z }, float3{ minCoord.x, minCoord.y, minCoord.z }, float3{ voxelSize.x, voxelSize.y, voxelSize.z }, cudaGridToMeshXf,
                                     data_->dipoles.data(), data_->cudaNodes.data(), data_->cudaMeshPoints.data(), data_->cudaFaces.data(), data_->cudaResult.data(), beta );   
    
    data_->cudaResult.toVector( res );
}

}
}