#pragma once

#include "MRVector3.h"

namespace MR
{
 
// 3-dimensional line: cross( x - p, d ) = 0
template <typename T> 
struct Line3
{
    Vector3<T> p, d;

    constexpr Line3() noexcept = default;
    constexpr Line3( const Vector3<T> & p, const Vector3<T> & d ) noexcept : p( p ), d( d ) { }
    template <typename U>
    constexpr explicit Line3( const Line3<U> & l ) noexcept : p( l.p ), d( l.d ) { }

    // returns squared distance from given point to this line
    [[nodiscard]] T distanceSq( const Vector3<T> & x ) const 
        { return ( x - project( x ) ).lengthSq(); }

    // returns same line represented with flipped direction of d-vector
    [[nodiscard]] Line3 operator -() const { return Line3( p, -d ); }
    // returns same representation
    [[nodiscard]] const Line3 & operator +() const { return *this; }
    // returns same line represented with unit d-vector
    [[nodiscard]] Line3 normalized() const { return { p, d.normalized() }; }

    // finds the closest point on line
    [[nodiscard]] Vector3<T> project( const Vector3<T> & x ) const { return p + dot( d, x - p ) / d.lengthSq() * d; }
};

// given line: l(x) = 0, and transformation: y=xf(x);
// returns the same line in y reference frame: l'(y) = 0;
// if given transformation is not rigid, then it is a good idea to normalize returned line
template <typename T>
[[nodiscard]] inline Line3<T> transformed( const Line3<T> & l, const AffineXf3<T> & xf )
{
    return Line3<T>{ xf( l.p ), xf.A * l.d };
}

template <typename T> 
[[nodiscard]] inline bool operator == ( const Line3<T> & a, const Line3<T> & b )
{
    return a.p == b.p && a.d == b.d;
}

template <typename T> 
[[nodiscard]] inline bool operator != ( const Line3<T> & a, const Line3<T> & b )
{
    return !( a == b );
}

} //namespace MR
