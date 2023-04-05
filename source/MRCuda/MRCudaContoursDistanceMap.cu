#include "MRCudaContoursDistanceMap.h"
#include "MRCudaBasic.h"
#include "MRMesh/MRAABBTreePolyline.h"
#include "device_launch_parameters.h"

#include "MRCudaFloat2.cuh"

namespace MR
{
namespace Cuda
{

struct Box2
{
    float2 min;
    float2 max;
};

struct Node
{
    Box2 box;
    int l, r;
};

__device__ bool leaf( const Node& node )
{
    return node.r < 0;
}

__device__ int leafId( const Node& node )
{
    return node.l;
}

__device__ float2 getBoxClosestPointTo( const Box2& box, const float2& pt )
{
    return { clamp( pt.x, box.min.x, box.max.x ), clamp( pt.y, box.min.y, box.max.y ) };
}

using HalfEdgeRecord = int2;

__device__ float2 closestPointOnLineSegm( const float2& pt, const float2& a, const float2& b )
{
    const auto ab = b - a;
    const auto dt = dot( pt - a, ab );
    const auto abLengthSq = lengthSq( ab );
    if ( dt <= 0 )
        return a;
    if ( dt >= abLengthSq )
        return b;
    auto ratio = dt / abLengthSq;
    return a * ( 1 - ratio ) + b * ratio;
}

__global__ void kernel( const float2 originPoint, const int2 resolution, const float2 pixelSize, const Node* nodes, const float2* polylinePoints, const HalfEdgeRecord* edges, float* dists, const size_t size )
{
    if ( size == 0 )
    {
        assert( false );
        return;
    }

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= size )
        return;

    float2 pt;

    size_t x = index % resolution.x;
    size_t y = index / resolution.x;

    pt.x = pixelSize.x * x + originPoint.x;
    pt.y = pixelSize.y * y + originPoint.y;

    float resDistSq = FLT_MAX;

    struct SubTask
    {
        int n;
        float distSq = 0;
    };

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.distSq < resDistSq )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&] ( int n )
    {
        return SubTask{ n, lengthSq( getBoxClosestPointTo( nodes[n].box, pt ) - pt ) };
    };

    addSubTask( getSubTask( 0 ) );

    while ( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
        const auto& node = nodes[s.n];
        if ( s.distSq >= resDistSq )
            continue;

        if ( leaf( node ) )
        {
            const auto lineId = leafId( node );
            float2 a = polylinePoints[edges[2 * lineId].y];
            float2 b = polylinePoints[edges[2 * lineId + 1].y];
            auto proj = closestPointOnLineSegm( pt, a, b );

            float distSq = lengthSq( proj - pt );
            if ( distSq < resDistSq )
            {
                resDistSq = distSq;
            }
            continue;
        }

        auto s1 = getSubTask( node.l );
        auto s2 = getSubTask( node.r );
        if ( s1.distSq < s2.distSq )
        {
            const auto temp = s1;
            s1 = s2;
            s2 = temp;
        }
        assert( s1.distSq >= s2.distSq );
        addSubTask( s1 ); // larger distance to look later
        addSubTask( s2 ); // smaller distance to look first
    }

    dists[index] = sqrt( resDistSq );
}

DistanceMap distanceMapFromContours( const MR::Polyline2& polyline, const ContourToDistanceMapParams& params )
{    
    const auto& tree = polyline.getAABBTree();
    const auto& nodes = tree.nodes();
    const auto& edges = polyline.topology.edges();

    cudaSetDevice( 0 );
    const size_t size = size_t( params.resolution.x ) * params.resolution.y;

    DynamicArray<float2> cudaPts;
    cudaPts.fromVector( polyline.points.vec_ );    

    DynamicArray<Node> cudaNodes;
    cudaNodes.fromVector( nodes.vec_ );

    DynamicArray<HalfEdgeRecord> cudaEdges;
    cudaEdges.fromVector( edges.vec_ );

    DynamicArray<float> cudaRes (size);

    int maxThreadsPerBlock = 0;
    cudaDeviceGetAttribute( &maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0 );
    int numBlocks = ( int( size ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;

    // kernel
    kernel << <numBlocks, maxThreadsPerBlock >> > ( { params.orgPoint.x + params.pixelSize.x * 0.5f, params.orgPoint.y + params.pixelSize.y * 0.5f }, { params.resolution.x, params.resolution.y }, { params.pixelSize.x, params.pixelSize.y }, cudaNodes.data(), cudaPts.data(), cudaEdges.data(), cudaRes.data(), size );
    DistanceMap res( params.resolution.x, params.resolution.y );
    std::vector<float> vec( size );
    cudaRes.toVector( vec );
    res.set( std::move( vec ) );

    return res;
}

}
}