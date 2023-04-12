#include "MRCudaMeshProject.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRAABBTree.h"
#include "MRCudaMeshProject.cuh"
#include <chrono>
namespace MR
{

namespace Cuda
{

struct MeshData
{
    DynamicArray<float3> cudaMeshPoints;
    DynamicArray<Node3> cudaNodes;
    DynamicArray<HalfEdgeRecord> cudaEdges;
    DynamicArray<int> cudaEdgePerFace;
};

MeshProjector::MeshProjector()
{
    meshData_ = std::make_shared<MeshData>();
}

void MeshProjector::updateMeshData( std::shared_ptr<const MR::Mesh> mesh )
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

std::vector<MR::MeshProjectionResult> MeshProjector::findProjections(
     const std::vector<Vector3f>& points,  const AffineXf3f* xf, const AffineXf3f* refXfPtr, float upDistLimitSq, float loDistLimitSq )
{
    cudaSetDevice( 0 );    
    
    const size_t size = points.size();
    DynamicArray<float3> cudaPoints( points );    
    DynamicArray<MeshProjectionResult> cudaRes( size );

    AutoPtr<CudaXf> cudaXfPtr = AutoPtr<CudaXf>( xf );
    AutoPtr<CudaXf> cudaRefXfPtr = AutoPtr<CudaXf>( refXfPtr );

    meshProjectionKernel( cudaPoints.data(), meshData_->cudaNodes.data(), meshData_->cudaMeshPoints.data(), meshData_->cudaEdges.data(), meshData_->cudaEdgePerFace.data(), cudaRes.data(), cudaXfPtr.get(), cudaRefXfPtr.get(), upDistLimitSq, loDistLimitSq, size );
    std::vector<MR::MeshProjectionResult> res;
    cudaRes.toVector( res );    
    return res;
}

}

}