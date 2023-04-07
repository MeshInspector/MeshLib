#include "MRCudaMeshProject.h"
#include "MRMesh/MRAABBTree.h"
#include "MRCudaMeshProject.cuh"

namespace MR
{

namespace Cuda
{

std::vector<MR::MeshProjectionResult> findProjections( 
    const std::vector<Vector3f>& points, const MR::Mesh& mesh, float upDistLimitSq, float loDistLimitSq )
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

    DynamicArray<float3> cudaMeshPoints;
    cudaMeshPoints.fromVector( meshPoints.vec_ );

    DynamicArray<Node3> cudaNodes;
    cudaNodes.fromVector( nodes.vec_ );

    DynamicArray<HalfEdgeRecord> cudaEdges;
    cudaEdges.fromVector( edges.vec_ );

    DynamicArray<int> cudaEdgePerFace;
    cudaEdgePerFace.fromVector( edgePerFace.vec_ );

    DynamicArray<MeshProjectionResult> cudaRes( size );

    meshProjectionKernel( cudaPoints.data(), cudaNodes.data(), cudaMeshPoints.data(), cudaEdges.data(), cudaEdgePerFace.data(), cudaRes.data(), upDistLimitSq, loDistLimitSq, size );
    
    std::vector<MR::MeshProjectionResult> res;
    cudaRes.toVector( res );
    return res;
}

}

}