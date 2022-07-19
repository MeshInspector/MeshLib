#pragma once
// triangle-related mathematical functions are here

#include "MRVector3.h"
#include <limits>
#include <cmath>

namespace MR
{

/// Aspect ratio of a triangle is the ratio of the circum-radius to twice its in-radius
/// \ingroup MathGroup
template<typename T>
T triangleAspectRatio( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c )
{
    const auto bc = ( c - b ).length();
    const auto ca = ( a - c ).length(); 
    const auto ab = ( b - a ).length();
    auto halfPerimeter = ( bc + ca + ab ) / 2;
    auto den = 8 * ( halfPerimeter - bc ) * ( halfPerimeter - ca ) * ( halfPerimeter - ab );
    if ( den <= 0 )
        return std::numeric_limits<T>::max();

    return bc * ca * ab / den;
}

/// Computes dihedral angle between leftNorm and rightNorm, with sign
/// \ingroup MathGroup
template <typename T>
T dihedralAngle( const Vector3<T>& leftNorm, const Vector3<T>& rightNorm, const Vector3<T>& edgeVec )
{
    auto edgeDir = edgeVec.normalized();
    auto sin = dot( edgeDir, cross( leftNorm, rightNorm ) );
    auto cos = dot( leftNorm, rightNorm );
    return std::atan2( sin, cos );
}

} // namespace MR
