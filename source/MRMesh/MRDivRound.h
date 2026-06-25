#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// computes division n/d with rounding of the result to the nearest integer, all signs of n and d are supported
/// https://stackoverflow.com/a/18067292/7325599
template <class T>
inline T divRound( T n, T d )
{
    return ((n < 0) == (d < 0)) ? ((n + d/2)/d) : ((n - d/2)/d);
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
