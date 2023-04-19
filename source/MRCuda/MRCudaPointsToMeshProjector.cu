#include "MRCudaPointsToMeshProjector.cuh"
#include "MRMesh/MRAABBTree.h"
#include "device_launch_parameters.h"

namespace MR { namespace Cuda {

__device__ float3 Matrix4::transform( const float3& pt ) const
{
    float3 res = { dot( x, pt ), dot( y, pt ), dot( z, pt ) };
    res = res + b;
    return res;
}

__device__ Box3 Matrix4::transform( const Box3& box ) const
{
    Box3 res;
    res.include( transform( float3{ box.min.x, box.min.y, box.min.z } ) );
    res.include( transform( float3{ box.min.x, box.min.y, box.max.z } ) );
    res.include( transform( float3{ box.min.x, box.max.y, box.min.z } ) );
    res.include( transform( float3{ box.min.x, box.max.y, box.max.z } ) );
    res.include( transform( float3{ box.max.x, box.min.y, box.min.z } ) );
    res.include( transform( float3{ box.max.x, box.min.y, box.max.z } ) );
    res.include( transform( float3{ box.max.x, box.max.y, box.min.z } ) );
    res.include( transform( float3{ box.max.x, box.max.y, box.max.z } ) );
    return res;
}

__device__ bool Node3::leaf() const
{
    return r < 0;
}

__device__ int Node3::leafId() const
{
    return l;
}

__device__ float3 Box3::getBoxClosestPointTo( const float3& pt ) const
{
    return { clamp( pt.x, min.x, max.x ), clamp( pt.y, min.y, max.y ), clamp( pt.z, min.z, max.z ) };
}

__device__ void Box3::include( const float3& pt )
{
    if ( pt.x < min.x ) min.x = pt.x;
    if ( pt.x > max.x ) max.x = pt.x;
    if ( pt.y < min.y ) min.y = pt.y;
    if ( pt.y > max.y ) max.y = pt.y;
    if ( pt.z < min.z ) min.z = pt.z;
    if ( pt.z > max.z ) max.z = pt.z;
}

struct ClosestPointRes
{
    float2 bary;
    float3 proj;
};

__device__ ClosestPointRes closestPointInTriangle( const float3& p, const float3& a, const float3& b, const float3& c )
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = p - a;

    const float d1 = dot( ab, ap );
    const float d2 = dot( ac, ap );
    if ( d1 <= 0 && d2 <= 0 )
        return { { 0, 0 }, a };

    const float3 bp = p - b;
    const float d3 = dot( ab, bp );
    const float d4 = dot( ac, bp );
    if ( d3 >= 0 && d4 <= d3 )
        return { { 1, 0 }, b };

    const float3 cp = p - c;
    const float d5 = dot( ab, cp );
    const float d6 = dot( ac, cp );
    if ( d6 >= 0 && d5 <= d6 )
        return { { 0, 1 }, c };

    const float vc = d1 * d4 - d3 * d2;
    if ( vc <= 0 && d1 >= 0 && d3 <= 0 )
    {
        const float v = d1 / ( d1 - d3 );
        return { { v, 0 }, a + ab * v };
    }

    const float vb = d5 * d2 - d1 * d6;
    if ( vb <= 0 && d6 <= 0 )
    {
        const float v = d2 / ( d2 - d6 );
        return { { 0, v }, a + ac * v };
    }

    const float va = d3 * d6 - d5 * d4;
    if ( va <= 0 )
    {
        const float v = ( d4 - d3 ) / ( ( d4 - d3 ) + ( d5 - d6 ) );
        return { { 1 - v, v }, b + ( c - b ) * v };
    }

    const float denom = 1 / ( va + vb + vc );
    const float v = vb * denom;
    const float w = vc * denom;
    return { { v, w }, a + ab * v + ac * w };
}

__global__ void kernel( const float3* points,
    const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
    MeshProjectionResult* resVec, const Matrix4 xf, const Matrix4 refXf, float upDistLimitSq, float loDistLimitSq, size_t size )
{
    if ( size == 0 )
    {
        assert( false );
        return;
    }

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= size )
        return;

    const auto pt = xf.isIdentity ? points[index] : xf.transform( points[index] );
    MeshProjectionResult res;
    res.distSq = upDistLimitSq;
    res.proj.faceId = -1;
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
        const auto box = refXf.isIdentity ? nodes[n].box : refXf.transform( nodes[n].box );
        float distSq = lengthSq( box.getBoxClosestPointTo( pt ) - pt );
        return SubTask{ n, distSq };
    };

    addSubTask( getSubTask( 0 ) );
    
    while ( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
        const auto& node = nodes[s.n];
        if ( s.distSq >= res.distSq )
            continue;

        if ( node.leaf() )
        {
            const auto face = node.leafId();
            const auto & vs = faces[face].verts;
            float3 a = meshPoints[vs[0]];
            float3 b = meshPoints[vs[1]];
            float3 c = meshPoints[vs[2]];

            if ( !refXf.isIdentity )
            {
                a = refXf.transform( a );
                b = refXf.transform( b );
                c = refXf.transform( c );
            }
            
            // compute the closest point in double-precision, because float might be not enough
            const auto closestPointRes = closestPointInTriangle( pt, a, b, c );

            float distSq = lengthSq( closestPointRes.proj - pt );
            if ( distSq < res.distSq )
            {
                res.distSq = distSq;
                res.proj.point = closestPointRes.proj;
                res.proj.faceId = face;
                res.tp = MeshTriPoint{ -1, closestPointRes.bary.x, closestPointRes.bary.y };
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
    resVec[index] = res;
}

void meshProjectionKernel( const float3* points, 
                           const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                           MeshProjectionResult* resVec, const Matrix4 xf, const Matrix4 refXf, float upDistLimitSq, float loDistLimitSq, size_t size )
{
    int maxThreadsPerBlock = 0;
    cudaDeviceGetAttribute( &maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0 );
    int numBlocks = ( int( size ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;
    kernel << <numBlocks, maxThreadsPerBlock >> > ( points, nodes, meshPoints, faces, resVec, xf, refXf, upDistLimitSq, loDistLimitSq, size );
}

}}

