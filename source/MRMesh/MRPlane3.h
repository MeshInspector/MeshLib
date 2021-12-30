#pragma once

#include "MRVector3.h"

namespace MR
{
 
// 3-dimensional plane: dot(n,x) - d = 0
template <typename T> 
struct Plane3
{
    Vector3<T> n;
    T d = 0;

    constexpr Plane3() noexcept = default;
    constexpr Plane3( const Vector3<T> & n, T d ) noexcept : n( n ), d( d ) { }
    template <typename U>
    constexpr explicit Plane3( const Plane3<U> & p ) noexcept : n( p.n ), d( T( p.d ) ) { }
    [[nodiscard]] constexpr static Plane3 fromDirAndPt( const Vector3<T> & n, const Vector3<T> & p ) { return { n, dot( n, p ) }; }

    // returns distance from given point to this plane (if n is a unit vector)
    [[nodiscard]] T distance( const Vector3<T> & x ) const { return dot( n, x ) - d; }

    // returns same plane represented with flipped direction of n-vector
    [[nodiscard]] Plane3 operator -() const { return Plane3( -n, -d ); }
    // returns same representation
    [[nodiscard]] const Plane3 & operator +() const { return *this; }
    // returns same plane represented with unit n-vector
    [[nodiscard]] Plane3 normalized() const 
    {
        const auto len = n.length();
        if ( len <= 0 )
            return {};
        const auto rlen = 1 / len;
        return Plane3{ rlen * n, rlen * d };
    }

    // finds the closest point on plane
    [[nodiscard]] Vector3<T> project( const Vector3<T> & p ) const { return p - distance( p ) / n.lengthSq() * n; }
};

// given plane: pl(x) = 0, and inverse transformation: y=ixf^-1(x);
// returns the same plane in y reference frame: pl'(y) = 0;
// if given transformation is not rigid, then it is a good idea to normalize returned plane
template <typename T>
[[nodiscard]] inline Plane3<T> invTransformed( const Plane3<T> & pl, const AffineXf3<T> & ixf )
{
    return Plane3<T>{ ixf.A.transposed() * pl.n, pl.d - dot( pl.n, ixf.b ) };
}

// given plane: pl(x) = 0, and transformation: y=xf(x);
// returns the same plane in y reference frame: pl'(y) = 0;
// if given transformation is not rigid, then it is a good idea to normalize returned plane
template <typename T>
[[nodiscard]] inline Plane3<T> transformed( const Plane3<T> & plane, const AffineXf3<T> & xf )
{
    return invTransformed( plane, xf.inverse() );
}

template <typename T> 
[[nodiscard]] inline bool operator == ( const Plane3<T> & a, const Plane3<T> & b )
{
    return a.n == b.n && a.d == b.d;
}

template <typename T> 
[[nodiscard]] inline bool operator != ( const Plane3<T> & a, const Plane3<T> & b )
{
    return !( a == b );
}

} //namespace MR
