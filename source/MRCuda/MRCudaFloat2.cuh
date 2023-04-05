#include "cuda_runtime.h"
namespace MR
{

namespace Cuda
{

__device__ float clamp( float x, float l, float u )
{
    return ( ( x < l ) ? l :
             ( ( x > u ) ? u : x ) );
};

__device__ float2 operator+( const float2& a, const float2& b )
{
    return { a.x + b.x, a.y + b.y };
}

__device__ float2 operator-( const float2& a, const float2& b )
{
    return { a.x - b.x, a.y - b.y };
}

__device__ float2 operator*( const float2& a, const float k )
{
    return { k * a.x , k * a.y };
}

__device__ float lengthSq( const float2& a )
{
    return a.x * a.x + a.y * a.y;
}

__device__ float dot( const float2& a, const float2& b )
{
    return a.x * b.x + a.y * b.y;
}


}
}