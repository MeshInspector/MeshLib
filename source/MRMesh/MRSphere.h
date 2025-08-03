#pragma once

#include "MRMeshFwd.h" // To fix `attribute declaration must precede definition` on `Sphere`.

namespace MR
{

/// \ingroup MathGroup
template <typename V>
struct Sphere
{
    using T = typename V::ValueType;

    V center;
    T radius = 0;

    constexpr Sphere() noexcept = default;
    constexpr Sphere( const V & c, T r ) noexcept : center( c ), radius( r ) { }
    template <typename U>
    constexpr explicit Sphere( const Sphere<U> & l ) noexcept : center( l.center ), radius( T( l.radius ) ) { }

    /// finds the closest point on sphere
    [[nodiscard]] V project( const V & x ) const { return center + radius * ( x - center ).normalized(); }

    /// returns signed distance from given point to this sphere:
    /// positive - outside, zero - on sphere, negative - inside
    [[nodiscard]] T distance( const V & x ) const { return ( x - center ).length() - radius; }

    /// returns squared distance from given point to this sphere
    [[nodiscard]] T distanceSq( const V & x ) const { return sqr( distance( x ) ); }

    friend bool operator == ( const Sphere & a, const Sphere & b ) = default;
};

} // namespace MR
