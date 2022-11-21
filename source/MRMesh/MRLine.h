#pragma once

namespace MR
{
 
/// 2- or 3-dimensional line: cross( x - p, d ) = 0
/// \ingroup MathGroup
template <typename V>
struct Line
{
    using T = typename V::ValueType;

    V p, d;

    constexpr Line() noexcept = default;
    constexpr Line( const V & p, const V & d ) noexcept : p( p ), d( d ) { }
    template <typename U>
    constexpr explicit Line( const Line<U> & l ) noexcept : p( l.p ), d( l.d ) { }

    /// returns squared distance from given point to this line
    [[nodiscard]] T distanceSq( const V & x ) const 
        { return ( x - project( x ) ).lengthSq(); }

    /// returns same line represented with flipped direction of d-vector
    [[nodiscard]] Line operator -() const { return Line( p, -d ); }
    /// returns same representation
    [[nodiscard]] const Line & operator +() const { return *this; }
    /// returns same line represented with unit d-vector
    [[nodiscard]] Line normalized() const { return { p, d.normalized() }; }

    /// finds the closest point on line
    [[nodiscard]] V project( const V & x ) const { return p + dot( d, x - p ) / d.lengthSq() * d; }
};

/// \related Line
/// \{

/// given line: l(x) = 0, and transformation: y=xf(x);
/// \return the same line in y reference frame: l'(y) = 0;
/// \details if given transformation is not rigid, then it is a good idea to normalize returned line
template <typename V>
[[nodiscard]] inline Line<V> transformed( const Line<V> & l, const AffineXf<V> & xf )
{
    return Line<V>{ xf( l.p ), xf.A * l.d };
}

template <typename V>
[[nodiscard]] inline bool operator == ( const Line<V> & a, const Line<V> & b )
{
    return a.p == b.p && a.d == b.d;
}

/// \}

} // namespace MR
