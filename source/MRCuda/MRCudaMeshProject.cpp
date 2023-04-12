#include "MRCudaMeshProject.h"
#include "MRMesh/MRAABBTree.h"
#include "MRCudaMeshProject.cuh"

namespace MR
{

namespace Cuda
{

std::vector<MR::MeshProjectionResult> findProjections( 
     const std::vector<Vector3f>& points, const MR::Mesh& mesh, const AffineXf3f* xf, const AffineXf3f* refXfPtr, float upDistLimitSq, float loDistLimitSq )
{
    const AABBTree& tree = mesh.getAABBTree();
    const auto& nodes = tree.nodes();
    const auto& meshPoints = mesh.points;
    const auto& edges = mesh.topology.edges();
    const auto& edgePerFace = mesh.topology.edgePerFace();

    cudaSetDevice( 0 );
    const size_t size = points.size();

    DynamicArray<float3> cudaPoints( points );
    DynamicArray<float3> cudaMeshPoints( meshPoints.vec_ );
    DynamicArray<Node3> cudaNodes( nodes.vec_ );
    DynamicArray<HalfEdgeRecord> cudaEdges( edges.vec_ );
    DynamicArray<int> cudaEdgePerFace( edgePerFace.vec_ );
    DynamicArray<MeshProjectionResult> cudaRes( size );

    AutoPtr<CudaXf> cudaXfPtr = AutoPtr<CudaXf>( xf );
    AutoPtr<CudaXf> cudaRefXfPtr = AutoPtr<CudaXf>( refXfPtr );

    meshProjectionKernel( cudaPoints.data(), cudaNodes.data(), cudaMeshPoints.data(), cudaEdges.data(), cudaEdgePerFace.data(), cudaRes.data(), cudaXfPtr.get(), cudaRefXfPtr.get(), upDistLimitSq, loDistLimitSq, size );
    std::vector<MR::MeshProjectionResult> res;
    cudaRes.toVector( res );

    return res;
}

}

}