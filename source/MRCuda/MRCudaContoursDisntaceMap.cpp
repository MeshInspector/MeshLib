#include "MRCudaContoursDistanceMap.h"
#include "MRMesh/MRAABBTreePolyline.h"
#include "MRMesh/MRParallelFor.h"
#include "MRCudaContoursDistanceMap.cuh"

namespace MR
{

namespace Cuda
{

DistanceMap distanceMapFromContours( const MR::Polyline2& polyline, const ContourToDistanceMapParams& params )
{
    const auto& tree = polyline.getAABBTree();
    const auto& nodes = tree.nodes();

    CUDA_EXEC( cudaSetDevice( 0 ) );
    const size_t size = size_t( params.resolution.x ) * params.resolution.y;

    DynamicArray<float2> cudaPts;
    cudaPts.fromVector( polyline.points.vec_ );

    DynamicArray<Node2> cudaNodes;
    cudaNodes.fromVector( nodes.vec_ );

    Vector<int, EdgeId> orgs( polyline.topology.edgeSize() );
    ParallelFor( orgs, [&]( EdgeId i )
    {
        orgs[i] = polyline.topology.org( i );
    } );

    DynamicArray<int> cudaOrgs;
    cudaOrgs.fromVector( orgs.vec_ );

    DynamicArray<float> cudaRes( size );

    // kernel
    contoursDistanceMapProjectionKernel( 
        { params.orgPoint.x + params.pixelSize.x * 0.5f, params.orgPoint.y + params.pixelSize.y * 0.5f }, 
        { params.resolution.x, params.resolution.y }, 
        { params.pixelSize.x, params.pixelSize.y }, 
        cudaNodes.data(), cudaPts.data(), cudaOrgs.data(), cudaRes.data(), size );
    CUDA_EXEC( cudaGetLastError() );

    DistanceMap res( params.resolution.x, params.resolution.y );
    std::vector<float> vec( size );
    cudaRes.toVector( vec );
    res.set( std::move( vec ) );

    return res;
}

size_t distanceMapFromContoursHeapBytes( const MR::Polyline2& polyline, const ContourToDistanceMapParams& params )
{
    /// cannot use polyline.heapBytes here because it has extra fields in topology and does not create AABBTree if it is not present
    return 
        polyline.points.heapBytes() + 
        polyline.getAABBTree().nodes().heapBytes() +
        polyline.topology.edgeSize() +
        size_t( params.resolution.x ) * params.resolution.y * sizeof( float );
}

}

}