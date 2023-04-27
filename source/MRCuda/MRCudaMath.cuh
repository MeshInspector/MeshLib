#pragma once
#include "exports.h"
#include "cuda_runtime.h"
#include "MRCudaFloat.cuh"

namespace MR
{
namespace Cuda
{
// struct simular to MR::Box3 with minimal needed functions
struct Box3
{
    float3 min;
    float3 max;

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

}
}