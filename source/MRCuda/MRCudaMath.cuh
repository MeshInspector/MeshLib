#pragma once

#include "exports.h"
#include "MRCuda.cuh"
#include "MRCudaFloat.cuh"
#include "MRCudaInplaceStack.cuh"

#include <cfloat>
#include <cstdint>

namespace MR
{
namespace Cuda
{

// struct simular to MR::Box2 with minimal needed functions
struct Box2
{
    float2 min;
    float2 max;

    __device__ float2 getBoxClosestPointTo( const float2& pt ) const
    {
        return { clamp( pt.x, min.x, max.x ), clamp( pt.y, min.y, max.y ) };
    }
};

// struct simular to MR::Box3 with minimal needed functions
struct Box3
{
    float3 min;
    float3 max;

    __device__ bool valid() const
    {
        return min.x <= max.x && min.y <= max.y && min.z <= max.z;
    }

    __device__ float3 getBoxClosestPointTo( const float3& pt ) const
    {
        return { clamp( pt.x, min.x, max.x ), clamp( pt.y, min.y, max.y ), clamp( pt.z, min.z, max.z ) };
    }

    __device__ void include( const float3& pt )
    {
        if ( pt.x < min.x ) min.x = pt.x;
        if ( pt.x > max.x ) max.x = pt.x;
        if ( pt.y < min.y ) min.y = pt.y;
        if ( pt.y > max.y ) max.y = pt.y;
        if ( pt.z < min.z ) min.z = pt.z;
        if ( pt.z > max.z ) max.z = pt.z;
    }

    __device__ Box3 intersection( const Box3& b ) const
    {
        Box3 res;
        res.min.x = fmax( min.x, b.min.x );
        res.min.y = fmax( min.y, b.min.y );
        res.min.z = fmax( min.z, b.min.z );
        res.max.x = fmin( max.x, b.max.x );
        res.max.y = fmin( max.y, b.max.y );
        res.max.z = fmin( max.z, b.max.z );
        return res;
    }

    __device__ float3 operator[]( const int i ) const
    {
        assert( i == 0 || i == 1 );
        return ( i == 0 ) ? min : max;
    }
};

struct IntersectionPrecomputes
{
    float3 dir;
    float3 invDir;
    int maxDimIdxZ = 2;
    int idxX = 0;
    int idxY = 1;
    int3 sign;
    float Sx, Sy, Sz;
};

// struct simular to AABBTreeNode<LineTreeTraits<Vector2f>> with minimal needed functions
struct Node2
{
    Box2 box;
    int l, r;

    __device__ bool leaf() const
    {
        return r < 0;
    }
    __device__ int leafId() const
    {
        return l;
    }
};

// struct simular to AABBTreeNode<FaceTreeTraits3> with minimal needed functions
struct Node3
{
    Box3 box;
    int l, r;

    __device__ bool leaf() const
    {
        return r < 0;
    }
    __device__ int leafId() const
    {
        return l;
    }
    __device__ int2 getLeafPointRange() const
    { 
        return { -(l + 1), -(r + 1) };
    }
};

struct FaceToThreeVerts
{
    int verts[3];
};

// GPU analog of CPU PointOnFace struct
struct PointOnFace
{
    int faceId;
    float3 point;
};

// GPU analog of CPU MeshTriPoint struct
struct MeshTriPoint
{
    int unused = -1; //always -1, but necessary to have the same size as MeshTriPoint in CPU
    float a;
    float b;
};

struct MeshIntersectionResult
{
    PointOnFace proj;
    MeshTriPoint tp;
    float distanceAlongLine = -FLT_MAX;
};

struct TriIntersectResult
{
    // barycentric representation
    float a;
    float b;
    // distance from ray origin to p in dir length units
    float t = -FLT_MAX;
};

// GPU analog of CPU AffineXf3f
struct Matrix4
{
    float3 x = { 1.f, 0.f, 0.f };
    float3 y = { 0.f, 1.f, 0.f };
    float3 z = { 0.f, 0.f, 1.f };
    float3 b = { 0.f, 0.f, 0.f };
    bool isIdentity = true;
    /// application of the transformation to a point
    __device__ float3 transform( const float3& pt ) const
    {
        float3 res = { dot( x, pt ), dot( y, pt ), dot( z, pt ) };
        res = res + b;
        return res;
    }
    /// application of the transformation to a box
    __device__ Box3 transform( const Box3& box ) const
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
};

struct ClosestPointRes
{
    float2 bary;
    float3 proj;
};

__device__ inline ClosestPointRes closestPointInTriangle( const float3& p, const float3& a, const float3& b, const float3& c )
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

__device__ inline bool rayBoxIntersect( const Box3& box, const float3& rayOrigin, float& t0, float& t1, const IntersectionPrecomputes& prec )
{
    const int3& sign = prec.sign;

    // compare and update x-dimension with t0-t1
    t1 = min( ( box[sign.x].x - rayOrigin.x ) * prec.invDir.x, t1 );
    t0 = max( ( box[1 - sign.x].x - rayOrigin.x ) * prec.invDir.x, t0 );

    // compare and update y-dimension with t0-t1
    t1 = min( ( box[sign.y].y - rayOrigin.y ) * prec.invDir.y, t1 );
    t0 = max( ( box[1 - sign.y].y - rayOrigin.y ) * prec.invDir.y, t0 );

    // compare and update z-dimension with t0-t1
    t1 = min( ( box[sign.z].z - rayOrigin.z ) * prec.invDir.z, t1 );
    t0 = max( ( box[1 - sign.z].z - rayOrigin.z ) * prec.invDir.z, t0 );
    return t0 <= t1;
}

__device__ inline TriIntersectResult rayTriangleIntersect(const float* oriA, const float* oriB, const float* oriC, const IntersectionPrecomputes& prec)
{
    const float Sx = prec.Sx;
    const float Sy = prec.Sy;
    const float Sz = prec.Sz;    

    const float Ax = oriA[prec.idxX] - Sx * oriA[prec.maxDimIdxZ];
    const float Ay = oriA[prec.idxY] - Sy * oriA[prec.maxDimIdxZ];
    const float Bx = oriB[prec.idxX] - Sx * oriB[prec.maxDimIdxZ];
    const float By = oriB[prec.idxY] - Sy * oriB[prec.maxDimIdxZ];
    const float Cx = oriC[prec.idxX] - Sx * oriC[prec.maxDimIdxZ];
    const float Cy = oriC[prec.idxY] - Sy * oriC[prec.maxDimIdxZ];

    // due to fused multiply-add (FMA): (A*B-A*B) can be different from zero, so we need epsilon
    const float eps = FLT_EPSILON * max( Ax, Bx, Cx, Ay, By, Cy );

    float U = Cx * By - Cy * Bx;
    float V = Ax * Cy - Ay * Cx;
    float W = Bx * Ay - By * Ax;
    TriIntersectResult res;
    if ( U < -eps || V < -eps || W < -eps )
    {
        if ( U > eps || V > eps || W > eps )
        {
            // U,V,W have clearly different signs, so the ray misses the triangle
            return res;
        }
    }

    float det = U + V + W;
    if ( det == 0 )
        return res;

    const float Az = Sz * oriA[prec.maxDimIdxZ];
    const float Bz = Sz * oriB[prec.maxDimIdxZ];
    const float Cz = Sz * oriC[prec.maxDimIdxZ];
    const float t = U * Az + V * Bz + W * Cz;
    
    res.a = V / det;
    res.b = W / det;
    res.t = t / det;
    return res;
}

__device__ inline MeshIntersectionResult rayMeshIntersect( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, 
                                                           const float3& rayOrigin, float rayStart, float rayEnd, const IntersectionPrecomputes& prec, bool closestIntersect = true )
{
    const Box3& box = nodes[0].box;
    MeshIntersectionResult res;
    res.distanceAlongLine = -FLT_MAX;

    float start = rayStart;
    float end = rayEnd;
    if ( !rayBoxIntersect( box, rayOrigin, start, end, prec ) )
        return res;

    struct SubTask
    {
        int n;
        float rayStart;
    };

    InplaceStack<SubTask, 32> subtasks;

    auto addSubTask = [&] ( int n, float rayStart )
    {
        subtasks.push( { n, rayStart } );
    };

    addSubTask( 0, rayStart );

    int faceId = -1;
    float baryA = 0;
    float baryB = 0;

    while ( !subtasks.empty() && ( closestIntersect || faceId < 0 ) )
    {
        const auto s = subtasks.top();
        subtasks.pop();

        if ( s.rayStart < rayEnd )
        {
            if ( nodes[s.n].leaf() )
            {
                auto face = nodes[s.n].leafId();
                const auto& vs = faces[face].verts;
                float3 a = meshPoints[vs[0]] - rayOrigin;
                float3 b = meshPoints[vs[1]] - rayOrigin;
                float3 c = meshPoints[vs[2]] - rayOrigin;
                
                const auto tri = rayTriangleIntersect( ( float* ) (&a), ( float* ) (&b), ( float* ) (&c), prec );
                if ( tri.t < rayEnd && tri.t > rayStart )
                {
                    faceId = face;
                    baryA = tri.a;
                    baryB = tri.b;
                    rayEnd = tri.t;
                }
            }
            else
            {
                float lStart = rayStart, lEnd = rayEnd;
                float rStart = rayStart, rEnd = rayEnd;

                if ( rayBoxIntersect( nodes[nodes[s.n].l].box, rayOrigin, lStart, lEnd, prec ) )
                {
                    if ( rayBoxIntersect( nodes[nodes[s.n].r].box, rayOrigin, rStart, rEnd, prec ) )
                    {
                        if ( lStart > rStart )
                        {
                            addSubTask( nodes[s.n].l, lStart );
                            addSubTask( nodes[s.n].r, rStart );
                        }
                        else
                        {
                            addSubTask( nodes[s.n].r, rStart );
                            addSubTask( nodes[s.n].l, lStart );
                        }
                    }
                    else
                    {
                        addSubTask( nodes[s.n].l, lStart );
                    }
                }
                else
                {
                    if ( rayBoxIntersect( nodes[nodes[s.n].r].box, rayOrigin, rStart, rEnd, prec ) )
                    {
                        addSubTask( nodes[s.n].r, rStart );
                    }
                }
            }
        }
    }
    if ( faceId < 0 )
        return res;

    res.proj.faceId = faceId;
    res.proj.point.x = rayOrigin.x + rayEnd * prec.dir.x;
    res.proj.point.y = rayOrigin.y + rayEnd * prec.dir.y;
    res.proj.point.z = rayOrigin.z + rayEnd * prec.dir.z;

    res.tp.a = baryA;
    res.tp.b = baryB;

    res.distanceAlongLine = rayEnd;
    return res;
}

__device__ inline bool testBit( const uint64_t* bitSet, const size_t bitNumber )
{
    return bool( ( bitSet[bitNumber / 64] >> ( bitNumber % 64 ) ) & 1 );
}

__device__ inline void setBit( uint64_t* bitSet, const size_t bitNumber )
{
    bitSet[bitNumber / 64] += ( 1ull << ( bitNumber % 64 ) );
}

template <typename T>
__device__ inline T sqr( T x )
{
    return x * x;
}

template <typename T>
__device__ inline int sgn( T x )
{
    return x > 0 ? 1 : ( x < 0 ? -1 : 0 );
}

}
}