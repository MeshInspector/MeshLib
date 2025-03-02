#pragma once

#include "MRMeshFwd.h"
#include "MRVectorTraits.h"

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

} //namespace MR
