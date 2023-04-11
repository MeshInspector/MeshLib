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

    DynamicArray<float3> cudaPoints;
    cudaPoints.fromVector( points );
    std::vector<float3> cudaPointsCopy;
    cudaPoints.toVector( cudaPointsCopy );

    DynamicArray<float3> cudaMeshPoints;
    cudaMeshPoints.fromVector( meshPoints.vec_ );
    std::vector<float3> cudaMeshPointsCopy;
    cudaMeshPoints.toVector( cudaMeshPointsCopy );

    DynamicArray<Node3> cudaNodes;
    cudaNodes.fromVector( nodes.vec_ );
    std::vector<Node3> cudaNodesCopy;
    cudaNodes.toVector( cudaNodesCopy );

    DynamicArray<HalfEdgeRecord> cudaEdges;
    cudaEdges.fromVector( edges.vec_ );
    std::vector<HalfEdgeRecord> cudaEdgesCopy;
    cudaEdges.toVector( cudaEdgesCopy );

    DynamicArray<int> cudaEdgePerFace;
    cudaEdgePerFace.fromVector( edgePerFace.vec_ );
    std::vector<int> cudaEdgerPerFaceCopy;
    cudaEdgePerFace.toVector( cudaEdgerPerFaceCopy );


    DynamicArray<MeshProjectionResult> cudaRes( size );
    std::vector< MeshProjectionResult > cudaResCopy( size );

    AutoPtr<CudaXf> cudaXfPtr = AutoPtr<CudaXf>( xf );
    AutoPtr<CudaXf> cudaRefXfPtr = AutoPtr<CudaXf>( refXfPtr );
    auto cudaXfPtrCopy = cudaXfPtr.convert<CudaXf>();
    auto cudaRefXfPtrCopy = cudaRefXfPtr.convert<CudaXf>();

    int maxThreadsPerBlock = 0;
    cudaDeviceGetAttribute( &maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0 );
    int numBlocks = ( int( size ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;

    meshProjectionKernel( cudaPoints.data(), cudaNodes.data(), cudaMeshPoints.data(), cudaEdges.data(), cudaEdgePerFace.data(), cudaRes.data(), cudaXfPtr.get(), cudaRefXfPtr.get(), upDistLimitSq, loDistLimitSq, size );
    std::vector<MR::MeshProjectionResult> res;
    cudaRes.toVector( res );

    return res;
}

}

}