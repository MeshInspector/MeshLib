#pragma once

#include "MRMeshFwd.h" // To fix `attribute declaration must precede definition` on `Torus3`.
#include "MRVector3.h"

namespace MR
{

/// a torus as a mathematical object: the surface swept by a circle of radius minorRadius,
/// whose center moves around the axis on the circle of radius majorRadius
/// \ingroup MathGroup
template <typename T>
struct Torus3
{
    Vector3<T> center;           ///< the center of the torus on its axis
    Vector3<T> axis{ 0, 0, 1 };  ///< unit direction of the axis of revolution
    T majorRadius = 0;           ///< the distance from the axis to the centers of the tube sections
    T minorRadius = 0;           ///< the radius of the tube

    constexpr Torus3() noexcept = default;
    constexpr Torus3( const Vector3<T>& c, const Vector3<T>& a, T major, T minor ) noexcept
        : center( c ), axis( a ), majorRadius( major ), minorRadius( minor ) {}

    // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
    //   when generating the bindings, and results in duplicate functions in C#.
    template <typename U> MR_REQUIRES_IF_SUPPORTED( !std::is_same_v<T, U> )
    constexpr explicit Torus3( const Torus3<U> & t ) noexcept : center( t.center ), axis( t.axis ), majorRadius( T( t.majorRadius ) ), minorRadius( T( t.minorRadius ) ) { }

    /// the center of the tube section closest to given point
    [[nodiscard]] Vector3<T> tubeCenter( const Vector3<T>& x ) const
    {
        const auto d = x - center;
        const auto radial = d - dot( d, axis ) * axis;
        const auto len = radial.length();
        return center + majorRadius * ( len > 0 ? radial / len : axis.perpendicular().first );
    }

    /// returns signed distance from given point to this torus: positive - outside of the tube, negative - inside
    [[nodiscard]] T distance( const Vector3<T>& x ) const { return ( x - tubeCenter( x ) ).length() - minorRadius; }

    /// unit normal of the torus in the point closest to given one, directed outside of the tube
    [[nodiscard]] Vector3<T> normal( const Vector3<T>& x ) const { return ( x - tubeCenter( x ) ).normalized(); }

    /// finds the closest point on the torus
    [[nodiscard]] Vector3<T> project( const Vector3<T>& x ) const
    {
        const auto tc = tubeCenter( x );
        return tc + minorRadius * ( x - tc ).normalized();
    }

    friend bool operator ==( const Torus3& a, const Torus3& b ) = default;
};

} // namespace MR
