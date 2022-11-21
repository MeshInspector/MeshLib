#pragma once

#include "MRMeshFwd.h"

namespace MR
{
 
// a segment of straight dimensional line
template <typename V> 
struct LineSegm
{
    using T = typename V::ValueType;
    V a, b;

    [[nodiscard]] constexpr LineSegm() noexcept = default;
    [[nodiscard]] constexpr LineSegm( const V & a, const V & b ) noexcept : a( a ), b( b ) { }
    template <typename U>
    [[nodiscard]] constexpr explicit LineSegm( const LineSegm<U> & p ) noexcept : a( p.a ), b( p.b ) { }
};

template <typename V> 
[[nodiscard]] inline bool operator == ( const LineSegm<V> & a, const LineSegm<V> & b )
{
    return a.a == b.a && a.b == b.b;
}

template <typename V> 
[[nodiscard]] V closestPointOnLineSegm( const V& pt, const LineSegm<V> & l )
{
    auto ab = l.b - l.a;
    auto dt = dot( pt - l.a, ab );
    auto abLengthSq = ab.lengthSq();
    if ( dt <= 0 )
        return l.a;
    if ( dt >= abLengthSq )
        return l.b;
    auto ratio = dt / abLengthSq;
    return l.a * ( 1 - ratio ) + l.b * ratio;
}

// returns true if two 2D segments intersect
template <typename V> 
[[nodiscard]] bool doSegmentsIntersect( const LineSegm<V> & x, const LineSegm<V> & y )
{
    auto xvec = x.b - x.a;
    if ( cross( xvec, y.a - x.a ) * cross( xvec, y.b - x.a ) > 0 )
        return false;
    auto yvec = y.b - y.a;
    if ( cross( yvec, x.a - y.a ) * cross( yvec, x.b - y.a ) > 0 )
        return false;
    return true;
}

} //namespace MR
