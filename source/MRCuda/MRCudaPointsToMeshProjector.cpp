#include "MRCudaPointsToMeshProjector.h"
#include "MRCudaPointsToMeshProjector.cuh"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRMatrix3Decompose.h"
#include "MRMesh/MRTimer.h"
#include <chrono>

namespace MR
{

namespace Cuda
{

struct MeshProjectorData
{
    DynamicArray<float3> cudaPoints;
    DynamicArray<MeshProjectionResult> cudaResult;
    DynamicArray<float3> cudaMeshPoints;
    DynamicArray<Node3> cudaNodes;
    DynamicArray<HalfEdgeRecord> cudaEdges;
    DynamicArray<int> cudaEdgePerFace;

    Matrix4 xf;
    Matrix4 refXf;
};

PointsToMeshProjector::PointsToMeshProjector()
{
    meshData_ = std::make_shared<MeshProjectorData>();
}

void PointsToMeshProjector::updateMeshData( std::shared_ptr<const MR::Mesh> mesh )
{
    if ( !mesh )
        return;

    const AABBTree& tree = mesh->getAABBTree();
    const auto& nodes = tree.nodes();
    const auto& meshPoints = mesh->points;
    const auto& edges = mesh->topology.edges();
    const auto& edgePerFace = mesh->topology.edgePerFace();

    meshData_->cudaMeshPoints.fromVector( meshPoints.vec_ );
    meshData_->cudaNodes.fromVector( nodes.vec_ );
    meshData_->cudaEdges.fromVector( edges.vec_ );
    meshData_->cudaEdgePerFace.fromVector( edgePerFace.vec_ );
}

void PointsToMeshProjector::findProjections(
      std::vector<MR::MeshProjectionResult>& res, const std::vector<Vector3f>& points, const AffineXf3f& objXf, const AffineXf3f& refObjXf, float upDistLimitSq, float loDistLimitSq )
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

    if ( !isRigid( refObjXf.A ) )
    {
        meshData_->refXf = getCudaMatrix( refObjXf );
        if ( objXf != AffineXf3f{} )
            meshData_->xf = getCudaMatrix( objXf );

        return;
    }

    const auto temp = refObjXf.inverse() * objXf;
    if ( temp != AffineXf3f{} )
        meshData_->xf = getCudaMatrix( temp );
    
    const size_t size = points.size();
    res.resize( size );
    meshData_->cudaPoints.fromVector( points );
    meshData_->cudaResult.resize( size );

    meshProjectionKernel( meshData_->cudaPoints.data(), meshData_->cudaNodes.data(), meshData_->cudaMeshPoints.data(), meshData_->cudaEdges.data(), meshData_->cudaEdgePerFace.data(), meshData_->cudaResult.data(), meshData_->xf, meshData_->refXf, upDistLimitSq, loDistLimitSq, size );
    meshData_->cudaResult.toVector( res );
}

}

}