#pragma once

#include "MRMeshFwd.h"
#include "MRVectorTraits.h"
#include <array>

namespace MR
{

/// Cubic Bezier curve
template <typename V>
struct CubicBezierCurve
{
    using VTraits = VectorTraits<V>;
    using T = typename VTraits::BaseType;
    static constexpr int elements = VTraits::size;

    /// 4 control points
    V p[4];

    /// computes point on the curve from parameter value
    V getPoint( T t ) const;

    /// computes weights of every control point for given parameter value, the sum of all weights is equal to 1
    static std::array<T, 4> getWeights( T t );
};

template <typename V>
inline V CubicBezierCurve<V>::getPoint( T t ) const
{
    // De Casteljau's algorithm
    V q[3];
    for ( int i = 0; i < 3; ++i )
        q[i] = lerp( p[i], p[i+1], t );

    V r[2];
    for ( int i = 0; i < 2; ++i )
        r[i] = lerp( q[i], q[i+1], t );

    return lerp( r[0], r[1], t );
}

template <typename V>
inline auto CubicBezierCurve<V>::getWeights( T t ) -> std::array<T, 4>
{
    T s = 1 - t;
    std::array<T, 4> w =
    {
            s * s * s,
        3 * s * s * t,
        3 * s * t * t
    };
    w[3] = 1 - w[0] - w[1] - w[2];
    return w;
}

} //namespace MR
