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
    /// returns directional vector of the line
    [[nodiscard]] V dir() const { return b - a; }
    /// returns squared length of this line segment
    [[nodiscard]] T lengthSq() const { return dir().lengthSq(); }
    /// returns the length of this line segment
    [[nodiscard]] T length() const { return dir().length(); }
    /// returns point on the line, where param=0 returns a and param=1 returns b
    [[nodiscard]] V operator()( T param ) const { return ( 1 - param ) * a + param * b; }
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

/// returns true if two 2D segments intersect,
/// optionally outputs intersection point as a parameter on both segments
template <typename V> 
[[nodiscard]] bool doSegmentsIntersect( const LineSegm<V> & x, const LineSegm<V> & y,
    typename V::ValueType * xPos = nullptr, typename V::ValueType * yPos = nullptr )
{
    // check whether infinite line x intersect segment y
    const auto xvec = x.b - x.a;
    const auto ya = cross( xvec, y.a - x.a );
    const auto yb = cross( xvec, y.b - x.a );
    if ( ya * yb > 0 )
        return false;

    // check whether infinite line y intersect segment x
    const auto yvec = y.b - y.a;
    const auto xa = cross( yvec, x.a - y.a );
    const auto xb = cross( yvec, x.b - y.a );
    if ( xa * xb > 0 )
        return false;

    if ( xPos )
    {
        // calculates intersection position on segment x
        const auto denom = xa - xb;
        *xPos = denom == 0 ? 0 : xa / denom;
    }
    if ( yPos )
    {
        // calculates intersection position on segment y
        const auto denom = ya - yb;
        *yPos = denom == 0 ? 0 : ya / denom;
    }
    return true;
}

/// returns true if 2D line segment and 2D line intersect,
/// optionally outputs intersection point as a parameter on both
template <typename V> 
[[nodiscard]] bool doSegmentLineIntersect( const LineSegm<V> & x, const Line<V> & y,
    typename V::ValueType * xPos = nullptr, typename V::ValueType * yPos = nullptr )
{
    // check whether infinite line y intersect segment x
    const auto xa = cross( y.d, x.a - y.p );
    const auto xb = cross( y.d, x.b - y.p );
    
    // if several line segments are connected in polyline, this ensures that only one of them will be intersected,
    // even is the line passes exactly through segment's end
    if ( ( xa <= 0 ) == ( xb <= 0 ) )
        return false; // both ends of the segment are at the same side of the line

    if ( xPos )
    {
        // calculates intersection position on segment x
        const auto denom = xa - xb;
        *xPos = denom == 0 ? 0 : xa / denom;
    }
    if ( yPos )
    {
        // calculates intersection position on line y
        const auto xvec = x.b - x.a;
        const auto ya = cross( xvec, y.p - x.a );
        const auto yb = cross( xvec, y.p + y.d - x.a );

        const auto denom = ya - yb;
        *yPos = denom == 0 ? 0 : ya / denom;
    }
    return true;
}

} //namespace MR
