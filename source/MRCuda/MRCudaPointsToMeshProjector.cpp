#include "MRCudaPointsToMeshProjector.h"
#include "MRCudaPointsToMeshProjector.cuh"

#include "MRCudaBasic.h"

#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRChunkIterator.h"
#include "MRMesh/MRMatrix3Decompose.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

namespace Cuda
{

struct MeshProjectorData
{
    DynamicArray<float3> cudaMeshPoints;
    DynamicArray<Node3> cudaNodes;
    DynamicArray<FaceToThreeVerts> cudaFaces;
};

PointsToMeshProjector::PointsToMeshProjector()
{
    meshData_ = std::make_shared<MeshProjectorData>();
}

void PointsToMeshProjector::updateMeshData( const MR::Mesh* mesh )
{
    if ( !mesh )
    {
        mesh_ = nullptr;
        return;
    }
    MR_TIMER

    const AABBTree& tree = mesh->getAABBTree();
    const auto& nodes = tree.nodes();
    const auto& meshPoints = mesh->points;
    const auto tris = mesh->topology.getTriangulation();

    meshData_->cudaMeshPoints.fromVector( meshPoints.vec_ );
    meshData_->cudaNodes.fromVector( nodes.vec_ );
    meshData_->cudaFaces.fromVector( tris.vec_ );

    mesh_ = mesh;
}

void PointsToMeshProjector::findProjections(
      std::vector<MR::MeshProjectionResult>& res, const std::vector<Vector3f>& points, const AffineXf3f* objXf, const AffineXf3f* refObjXf, float upDistLimitSq, float loDistLimitSq )
{
    MR_TIMER
    if ( !mesh_ )
    {
        assert( false );
        return;
    }
    
    CUDA_EXEC( cudaSetDevice( 0 ) );

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

    const AffineXf3f* notRigidRefXf{ nullptr };
    if ( refObjXf && !isRigid( refObjXf->A ) )
        notRigidRefXf = refObjXf;

    AffineXf3f xf;
    const AffineXf3f* xfPtr{ nullptr };
    if ( notRigidRefXf || !refObjXf )
        xfPtr = objXf;
    else
    {
        xf = refObjXf->inverse();
        if ( objXf )
            xf = xf * ( *objXf );
        xfPtr = &xf;
    }

    Matrix4 cudaRefXf;
    Matrix4 cudaXf;
    if ( notRigidRefXf )
        cudaRefXf = getCudaMatrix( *notRigidRefXf );
    if ( xfPtr )
        cudaXf = getCudaMatrix( *xfPtr );

    const auto totalSize = points.size();
    const auto bufferSize = maxBufferSize( getCudaSafeMemoryLimit(), totalSize, sizeof( float3 ) + sizeof( MeshProjectionResult ) );

    DynamicArray<float3> cudaPoints;
    cudaPoints.resize( bufferSize );

    DynamicArray<MeshProjectionResult> cudaResult;
    cudaResult.resize( bufferSize );
    res.resize( totalSize );

    for ( const auto [offset, size] : splitByChunks( totalSize, bufferSize ) )
    {
        cudaPoints.copyFrom( points.data() + offset, size );

        meshProjectionKernel( cudaPoints.data(), meshData_->cudaNodes.data(), meshData_->cudaMeshPoints.data(), meshData_->cudaFaces.data(), cudaResult.data(), cudaXf, cudaRefXf, upDistLimitSq, loDistLimitSq, size );
        CUDA_EXEC( cudaGetLastError() );

        cudaResult.copyTo( res.data() + offset, size );
    }

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, res.size() ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            if ( res[i].proj.face )
                res[i].mtp.e = mesh_->topology.edgeWithLeft( res[i].proj.face );
            else
                assert( !res[i].mtp.e );
        }
    } );
}

size_t PointsToMeshProjector::projectionsHeapBytes( size_t numProjections ) const
{
    return numProjections * ( sizeof( float3 ) + sizeof( MeshProjectionResult ) );
}

}

}