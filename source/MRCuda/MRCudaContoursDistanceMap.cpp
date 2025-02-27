#include "MRCudaContoursDistanceMap.h"
#include "MRCudaContoursDistanceMap.cuh"

#include "MRCudaBasic.h"

#include "MRMesh/MRAABBTreePolyline.h"
#include "MRMesh/MRChunkIterator.h"
#include "MRMesh/MRParallelFor.h"

namespace MR::Cuda
{

Expected<DistanceMap> distanceMapFromContours( const Polyline2& polyline, const ContourToDistanceMapParams& params )
{
    const auto& tree = polyline.getAABBTree();
    const auto& nodes = tree.nodes();

    CUDA_LOGE_RETURN_UNEXPECTED( cudaSetDevice( 0 ) );

    DynamicArray<float2> cudaPts;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaPts.fromVector( polyline.points.vec_ ) );

    DynamicArray<Node2> cudaNodes;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaNodes.fromVector( nodes.vec_ ) );

    Vector<int, EdgeId> orgs( polyline.topology.edgeSize() );
    ParallelFor( orgs, [&]( EdgeId i )
    {
        orgs[i] = polyline.topology.org( i );
    } );

    DynamicArray<int> cudaOrgs;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaOrgs.fromVector( orgs.vec_ ) );

    const auto totalSize = (size_t)params.resolution.x * params.resolution.y;
    const auto bufferSize = maxBufferSizeAlignedByBlock( getCudaSafeMemoryLimit(), params.resolution, sizeof( float ) );

    DynamicArrayF cudaRes;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaRes.resize( bufferSize ) );

    std::vector<float> vec( totalSize );

    for ( const auto [offset, size] : splitByChunks( totalSize, bufferSize ) )
    {
        // kernel
        contoursDistanceMapProjectionKernel(
            { params.orgPoint.x + params.pixelSize.x * 0.5f, params.orgPoint.y + params.pixelSize.y * 0.5f },
            { params.resolution.x, params.resolution.y },
            { params.pixelSize.x, params.pixelSize.y },
            cudaNodes.data(), cudaPts.data(), cudaOrgs.data(), cudaRes.data(), size, offset );
        CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );

        CUDA_LOGE_RETURN_UNEXPECTED( cudaRes.copyTo( vec.data() + offset, size ) );
    }

    DistanceMap res( params.resolution.x, params.resolution.y );
    res.set( std::move( vec ) );
    return res;
}

size_t distanceMapFromContoursHeapBytes( const Polyline2& polyline, const ContourToDistanceMapParams& params )
{
    constexpr size_t cMinRowCount = 10;
    /// cannot use polyline.heapBytes here because it has extra fields in topology and does not create AABBTree if it is not present
    return 
        polyline.points.heapBytes() + 
        polyline.getAABBTree().nodes().heapBytes() +
        polyline.topology.edgeSize() +
        cMinRowCount * params.resolution.y * sizeof( float );
}

} // namespace MR::Cuda
