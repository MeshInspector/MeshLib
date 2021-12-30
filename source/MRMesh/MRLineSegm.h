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

template <typename T> 
[[nodiscard]] inline bool operator == ( const LineSegm<T> & a, const LineSegm<T> & b )
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

} //namespace MR
