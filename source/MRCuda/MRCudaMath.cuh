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

}
}