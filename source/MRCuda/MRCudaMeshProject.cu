#include "MRCudaMeshProject.h"
#include "MRCudaBasic.h"
#include "MRMesh/MRAABBTree.h"
#include "device_launch_parameters.h"
#include "MRCudaFloat.cuh"

namespace MR { namespace Cuda {

struct Box3
{
    float3 min;
    float3 max;
};

struct Node3
{
    Box3 box;
    int l, r;
};

struct HalfEdgeRecord
{
    int next;
    int prev;
    int org;
    int left;
};

struct PointOnFace
{
    int faceId;
    float3 point;
};

struct MeshTriPoint
{
    int edgeId;
    float a;
    float b;
};

struct MeshProjectionResult
{
    PointOnFace proj;
    MeshTriPoint mtp;
    float distSq;
};

__device__ bool leaf( const Node3& node )
{
    return node.r < 0;
}

__device__ int leafId( const Node3& node )
{
    return node.l;
}

__device__ float3 getBoxClosestPointTo( const Box3& box, const float3& pt )
{
    return { clamp( pt.x, box.min.x, box.max.x ), clamp( pt.y, box.min.y, box.max.y ), clamp( pt.z, box.min.z, box.max.z ) };
}

struct ClosestPointRes
{
    float3 proj;
    float2 bary;
};

__device__ ClosestPointRes closestPointInTriangle( const float3& p, const float3& a, const float3& b, const float3& c )
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = p - a;

    const float d1 = dot( ab, ap );
    const float d2 = dot( ac, ap );
    if ( d1 <= 0 && d2 <= 0 )
        return { a, { 0, 0 } };

    const float3 bp = p - b;
    const float d3 = dot( ab, bp );
    const float d4 = dot( ac, bp );
    if ( d3 >= 0 && d4 <= d3 )
        return { b, { 1, 0 } };

    const float3 cp = p - c;
    const float d5 = dot( ab, cp );
    const float d6 = dot( ac, cp );
    if ( d6 >= 0 && d5 <= d6 )
        return { c, { 0, 1 } };

    const float vc = d1 * d4 - d3 * d2;
    if ( vc <= 0 && d1 >= 0 && d3 <= 0 )
    {
        const float v = d1 / ( d1 - d3 );
        return { a + ab * v, { v, 0 } };
    }

    const float vb = d5 * d2 - d1 * d6;
    if ( vb <= 0 && d2 >= 0 && d6 <= 0 )
    {
        const float v = d2 / ( d2 - d6 );
        return { a + ac * v, { 0, v } };
    }

    const float va = d3 * d6 - d5 * d4;
    if ( va <= 0 && ( d4 - d3 ) >= 0 && ( d5 - d6 ) >= 0 )
    {
        const float v = ( d4 - d3 ) / ( ( d4 - d3 ) + ( d5 - d6 ) );
        return { b + ( c - b ) * v, { 1 - v, v } };
    }

    const float denom = 1 / ( va + vb + vc );
    const float v = vb * denom;
    const float w = vc * denom;
    return { a + ab * v + ac * w, { v, w } };
}

__global__ void kernel( const float3* points, const Node3* nodes, const float3* meshPoints, const HalfEdgeRecord* edges, const int* edgePerFace, MeshProjectionResult* resVec, float upDistLimitSq, float loDistLimitSq, size_t size )
{
    if ( size == 0 )
    {
        assert( false );
        return;
    }

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= size )
        return;

    const auto& pt = points[index];
    auto& res = resVec[index];
   
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
        if ( s.distSq < res.distSq )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&] ( int n )
    {
        float distSq = lengthSq( getBoxClosestPointTo( nodes[n].box, pt ) - pt );
        return SubTask{ n, distSq };
    };

    addSubTask( getSubTask( 0 ) );
    
    while ( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
        const auto& node = nodes[s.n];
        if ( s.distSq >= res.distSq )
            continue;

        if ( leaf( node ) )
        {
            const auto face = leafId( node );
           
            int edge = edgePerFace[face];
            float3 a = meshPoints[edges[edge].org];
            edge = edges[ edge ^ 1 ].prev;
            float3 b = meshPoints[edges[edge].org];
            edge = edges[edge ^ 1].prev;
            float3 c = meshPoints[edges[edge].org];
            
            // compute the closest point in double-precision, because float might be not enough
            const auto closestPointRes = closestPointInTriangle( pt, a, b, c );

            float distSq = lengthSq( closestPointRes.proj - pt );
            if ( distSq < res.distSq )
            {
                res.distSq = distSq;
                res.proj.point = closestPointRes.proj;
                res.proj.faceId = face;
                res.mtp = MeshTriPoint{ edgePerFace[face], closestPointRes.bary.x, closestPointRes.bary.y };
                if ( distSq <= loDistLimitSq )
                    break;
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
}

std::vector<MR::MeshProjectionResult> findProjections( const std::vector<Vector3f>& points, const MR::Mesh& mesh, float upDistLimitSq, float loDistLimitSq )
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

    int maxThreadsPerBlock = 0;
    cudaDeviceGetAttribute( &maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0 );
    int numBlocks = ( int( size ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;

    kernel << <numBlocks, maxThreadsPerBlock >> > ( cudaPoints.data(), cudaNodes.data(), cudaMeshPoints.data(), cudaEdges.data(), cudaEdgePerFace.data(), cudaRes.data(), upDistLimitSq, loDistLimitSq, size );
    std::vector<MR::MeshProjectionResult> res;
    cudaRes.toVector( res );
    return res;
}

}}

