#pragma once
// triangle-related mathematical functions are here

#include "MRVector3.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace MR
{

/// Computes the squared diameter of the triangle's ABC circumcircle;
/// \ingroup MathGroup
template <typename T>
T circumcircleDiameterSq( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c )
{
    const auto ab = ( b - a ).lengthSq();
    const auto ca = ( a - c ).lengthSq();
    const auto bc = ( c - b ).lengthSq();
    if ( ab <= 0 )
        return ca;
    if ( ca <= 0 )
        return bc;
    if ( bc <= 0 )
        return ab;
    const auto f = cross( b - a, c - a ).lengthSq();
    if ( f <= 0 )
        return std::numeric_limits<T>::infinity();
    return ab * ca * bc / f;
}

/// Computes the diameter of the triangle's ABC circumcircle
/// \ingroup MathGroup
template <typename T>
inline T circumcircleDiameter( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c )
{
    return std::sqrt( circumcircleDiameterSq( a, b, c ) );
}

/// Computes sine of minimal angle in ABC triangle, which is equal to ratio of minimal edge length to circumcircle diameter
/// \ingroup MathGroup
template <typename T>
T minTriangleAngleSin( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c )
{
    const auto ab = ( b - a ).length();
    const auto ca = ( a - c ).length();
    const auto bc = ( c - b ).length();
    if ( ab <= 0  || ca <= 0 || bc <= 0 )
        return 0;
    const auto f = cross( b - a, c - a ).length();
    return f * std::min( { ab, ca, bc } ) / ( ab * ca * bc );
}

template <typename T>
T minTriangleAngle( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c )
{
    return std::asin( minTriangleAngleSin( a, b, c ) );
}

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

/// given an edge direction between two faces with given normals, computes sine of dihedral angle between the faces:
/// 0 if both faces are in the same plane,
/// positive if the faces form convex surface,
/// negative if the faces form concave surface
/// \ingroup MathGroup
template <typename T>
T dihedralAngleSin( const Vector3<T>& leftNorm, const Vector3<T>& rightNorm, const Vector3<T>& edgeVec )
{
    auto edgeDir = edgeVec.normalized();
    return dot( edgeDir, cross( leftNorm, rightNorm ) );
}

/// given two face normals, computes cosine of dihedral angle between the faces:
/// 1 if both faces are in the same plane,
/// 0 if the surface makes right angle turn at the edge,
/// -1 if the faces overlap one another
/// \ingroup MathGroup
template <typename T>
T dihedralAngleCos( const Vector3<T>& leftNorm, const Vector3<T>& rightNorm )
{
    return dot( leftNorm, rightNorm );
}

/// given an edge direction between two faces with given normals (not necessary of unit length), computes the dihedral angle between the faces:
/// 0 if both faces are in the same plane,
/// positive if the faces form convex surface,
/// negative if the faces form concave surface;
/// please consider the usage of faster dihedralAngleSin(e) and dihedralAngleCos(e)
/// \ingroup MathGroup
template <typename T>
T dihedralAngle( const Vector3<T>& leftNorm, const Vector3<T>& rightNorm, const Vector3<T>& edgeVec )
{
    auto sin = dihedralAngleSin( leftNorm, rightNorm, edgeVec );
    auto cos = dihedralAngleCos( leftNorm, rightNorm );
    return std::atan2( sin, cos );
}

/// given the lengths of 3 edges of triangle ABC, and
/// assuming that point B has coordinates (0,0); point A - (0,c);
/// computes the coordinates of point C (where c.x >= 0) or returns std::nullopt if input lengths are invalid for a triangle
template <typename T>
std::optional<Vector2<T>> posFromTriEdgeLengths( T a, T b, T c )
{
    if ( c == 0 )
    {
        if ( a == b )
            return Vector2<T>{ a, 0 };
        else
            return {};
    }
    const auto aa = sqr( a );
    const auto y = ( aa - sqr( b ) + sqr( c ) ) / ( 2 * c );
    const auto yy = sqr( y );
    if ( yy > aa )
        return {};
    const auto x = std::sqrt( aa - yy );
    return Vector2<T>{ x, y };
}

} // namespace MR
