#pragma once

#include "MRCuda.cuh"

#include <cassert>
#include <cmath>

namespace MR
{

namespace Cuda
{

__device__ inline float min( float x, float y )
{
    return ( ( x < y ) ? x : y );
}

template<typename T> __device__
 T max( T x, T y )
{
    return ( ( x > y ) ? x : y );
}

template <typename T, typename ... Args>  __device__
T max( T x, Args ... args )
{
    return max( x, max( args ... ) );
}

__device__ inline float clamp( float x, float l, float u )
{
    return ( ( x < l ) ? l :
             ( ( x > u ) ? u : x ) );
}

__device__ inline float2 operator+( const float2& a, const float2& b )
{
    return { a.x + b.x, a.y + b.y };
}

__device__ inline float2 operator-( const float2& a, const float2& b )
{
    return { a.x - b.x, a.y - b.y };
}

__device__ inline float2 operator*( const float2& a, const float k )
{
    return { k * a.x , k * a.y };
}

__device__ inline float lengthSq( const float2& a )
{
    return a.x * a.x + a.y * a.y;
}

__device__ inline float length( const float2& a )
{
    return sqrt( a.x * a.x + a.y * a.y );
}

__device__ inline float dot( const float2& a, const float2& b )
{
    return a.x * b.x + a.y * b.y;
}

__device__ inline float cross( const float2& a, const float2& b )
{
    return a.x * b.y - a.y * b.x;
}

__device__ inline float3 operator+( const float3& a, const float3& b )
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__device__ inline float3 operator-( const float3& a, const float3& b )
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

__device__ inline float3 operator*( const float3& a, const float k )
{
    return { k * a.x , k * a.y, k * a.z };
}

__device__ __host__ inline float3 operator/( const float3& a, const float k )
{
    return { a.x / k , a.y / k, a.z / k };
}

__device__ inline float lengthSq( const float3& a )
{
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__device__ inline float length( const float3& a )
{
    return sqrt( a.x * a.x + a.y * a.y + a.z * a.z );
}


__device__ inline float dot( const float3& a, const float3& b )
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 cross( const float3& a, const float3& b )
{
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

__device__ inline float3 normalize( const float3& v )
{
    float invLen = 1.0f / sqrtf( lengthSq( v ) );
    return v * invLen;
}

} //namespace Cuda

} //namespace MR
