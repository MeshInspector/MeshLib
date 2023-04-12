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

void MeshProjector::updateMeshData( std::shared_ptr<const MR::Mesh> mesh, std::string& log )
{
    if ( !mesh )
        return;

    const AABBTree& tree = mesh->getAABBTree();
    const auto& nodes = tree.nodes();
    const auto& meshPoints = mesh->points;
    const auto& edges = mesh->topology.edges();
    const auto& edgePerFace = mesh->topology.edgePerFace();

    const auto t0 = std::chrono::steady_clock::now();
    cudaDeviceSynchronize();
    meshData_->cudaMeshPoints.fromVector( meshPoints.vec_ );
    meshData_->cudaNodes.fromVector( nodes.vec_ );
    meshData_->cudaEdges.fromVector( edges.vec_ );
    meshData_->cudaEdgePerFace.fromVector( edgePerFace.vec_ );
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();
    log = ( std::string( "updateMeshData" ) + std::to_string( std::chrono::duration_cast< std::chrono::microseconds >( t1 - t0 ).count() ) + " mcs" );
}

std::vector<MR::MeshProjectionResult> MeshProjector::findProjections(
     const std::vector<Vector3f>& points,  const AffineXf3f* xf, const AffineXf3f* refXfPtr, float upDistLimitSq, float loDistLimitSq, std::vector<std::string>& log )
{
    const auto t0 = std::chrono::steady_clock::now();
       
    cudaSetDevice( 0 );
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const size_t size = points.size();
    DynamicArray<float3> cudaPoints( points );    
    DynamicArray<MeshProjectionResult> cudaRes( size );

    AutoPtr<CudaXf> cudaXfPtr = AutoPtr<CudaXf>( xf );
    AutoPtr<CudaXf> cudaRefXfPtr = AutoPtr<CudaXf>( refXfPtr );

    cudaDeviceSynchronize();
    const auto t2 = std::chrono::steady_clock::now();
    meshProjectionKernel( cudaPoints.data(), meshData_->cudaNodes.data(), meshData_->cudaMeshPoints.data(), meshData_->cudaEdges.data(), meshData_->cudaEdgePerFace.data(), cudaRes.data(), cudaXfPtr.get(), cudaRefXfPtr.get(), upDistLimitSq, loDistLimitSq, size );
    cudaDeviceSynchronize();
    const auto t3 = std::chrono::steady_clock::now();
    std::vector<MR::MeshProjectionResult> res;
    cudaRes.toVector( res );    
    cudaDeviceSynchronize();
    const auto t4 = std::chrono::steady_clock::now();

    log.push_back( std::string( "cudaSetDevice elapsed " ) + std::to_string( std::chrono::duration_cast< std::chrono::microseconds >( t1 - t0 ).count() )+ " mcs" );
    log.push_back( std::string( "copy data from host to device elapsed " ) + std::to_string( std::chrono::duration_cast< std::chrono::microseconds >( t2 - t1 ).count() ) + " mcs" );
    log.push_back( std::string( "kernel elapsed " ) + std::to_string( std::chrono::duration_cast< std::chrono::microseconds >( t3 - t2 ).count() ) + " mcs" );
    log.push_back( std::string( "copy data from device to host elapsed " ) + std::to_string( std::chrono::duration_cast< std::chrono::microseconds >( t4 - t3 ).count() ) + " mcs" );
    log.push_back( std::string( "With CUDA elapsed in total " ) + std::to_string( std::chrono::duration_cast< std::chrono::microseconds >( t4 - t0 ).count() ) + " mcs" );
    return res;
}

}

}