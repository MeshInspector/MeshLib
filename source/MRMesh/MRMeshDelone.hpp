#pragma once
#include <limits>

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

}
