#pragma once
#include <limits>
#include <cmath>

namespace MR
{

// computes the diameter of the triangle's ABC circumcircle
template <typename T>
T circumcircleDiameter( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c )
{
    const auto ab = ( b - a ).length();
    const auto ca = ( a - c ).length();
    const auto bc = ( c - b ).length();
    if ( ab <= 0 )
        return ca;
    if ( ca <= 0 )
        return bc;
    if ( bc <= 0 )
        return ab;
    const auto f = cross( b - a, c - a ).length();
    if ( f <= 0 )
        return std::numeric_limits<T>::max();
    return ab * ca * bc / f;
}

template <typename T>
T dihedralAngle( const Vector3<T>& leftNorm, const Vector3<T>& rightNorm, const Vector3<T>& edgeVec )
{
    auto edgeDir = edgeVec.normalized();
    auto sin = dot( edgeDir, cross( leftNorm, rightNorm ) );
    auto cos = dot( leftNorm, rightNorm );
    return std::atan2( sin, cos );
}

}
