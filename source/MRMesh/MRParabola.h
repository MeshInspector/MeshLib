#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// Represents quadratic function f(x) = a*x*x + b*x + c
template <typename T>
struct Parabola
{
    T a = 0;
    T b = 0;
    T c = 0;

    constexpr Parabola() noexcept = default;
    constexpr Parabola( T a, T b, T c ) : a( a ), b( b ), c( c ) { }
    template <typename U>
    constexpr explicit Parabola( const Parabola<U> & p ) : a( T( p.a ) ), b( T( p.b ) ), c( T( p.c ) ) { }

    /// compute value of quadratic function at any x
    constexpr T operator() ( T x ) const { return a*x*x + b*x + c; }
};

} //namespace MR
