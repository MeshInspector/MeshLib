#pragma once

#include "MRCudaMath.cuh"

namespace MR::Cuda
{

/// CUDA variant of MR::closestPointOnLineSegm( const MR::Vector2f&, const LineSegm2f& )
__device__ inline float2 closestPointOnLineSegm( const float2& pt, const float2& a, const float2& b )
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

/// CUDA variant of MR::closestPointOnLineSegm( const MR::Vector3f&, const LineSegm3f& )
__device__ inline float3 closestPointOnLineSegm( const float3& pt, const float3& a, const float3& b )
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

} // namespace MR::Cuda
