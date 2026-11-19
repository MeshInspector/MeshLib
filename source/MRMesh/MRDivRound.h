#pragma once

#include "MRMeshFwd.h"
#include "MRFastInt128.h"

namespace MR
{

/// computes division n/d with rounding of the result to the nearest integer, all signs of n and d are supported
/// https://stackoverflow.com/a/18067292/7325599
template <class T>
inline T divRound( T n, T d )
{
    return ((n < 0) == (d < 0)) ? ((n + d/2)/d) : ((n - d/2)/d);
}

/// the same for FastInt128, divided by divFastInt128, since its built-in division is unavailable in clang-cl builds
MR_BIND_IGNORE inline FastInt128 divRound( FastInt128 n, FastInt128 d )
{
    const FastInt128 h = d / 2; // by a constant, so without a call
    return divFastInt128( ( n < 0 ) == ( d < 0 ) ? n + h : n - h, d );
}

/// computes division n/d with rounding of each components to the nearest integer, all signs of n and d are supported
template <class T>
Vector2<T> divRound( const Vector2<T>& n, T d )
{
    return
    {
        divRound( n.x, d ),
        divRound( n.y, d )
    };
}

/// computes division n/d with rounding of each components to the nearest integer, all signs of n and d are supported
template <class T>
Vector3<T> divRound( const Vector3<T>& n, T d )
{
    return
    {
        divRound( n.x, d ),
        divRound( n.y, d ),
        divRound( n.z, d )
    };
}

/// computes division n/d with rounding of each components to the nearest integer, all signs of n and d are supported
template <class T>
Vector4<T> divRound( const Vector4<T>& n, T d )
{
    return
    {
        divRound( n.x, d ),
        divRound( n.y, d ),
        divRound( n.z, d ),
        divRound( n.w, d )
    };
}

} //namespace MR
