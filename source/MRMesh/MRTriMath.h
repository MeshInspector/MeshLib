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
[[nodiscard]] T circumcircleDiameterSq( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c )
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
[[nodiscard]] inline T circumcircleDiameter( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c )
{
    return std::sqrt( circumcircleDiameterSq( a, b, c ) );
}

/// Computes the center of the the triangle's 0AB circumcircle
template <typename T>
[[nodiscard]] Vector3<T> circumcircleCenter( const Vector3<T> & a, const Vector3<T> & b )
{
    const auto xabSq = cross( a, b ).lengthSq();
    const auto aa = a.lengthSq();
    const auto bb = b.lengthSq();
    if ( xabSq <= 0 )
    {
        if ( aa <= 0 )
            return b / T(2);
        // else b == 0 || a == b
        return a / T(2);
    }
    const auto ab = dot( a, b );
    return ( bb * ( aa - ab ) * a + aa * ( bb - ab ) * b ) / ( 2 * xabSq );
}

/// Computes the center of the the triangle's ABC circumcircle
template <typename T>
[[nodiscard]] inline Vector3<T> circumcircleCenter( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c )
{
    return circumcircleCenter( a - c, b - c ) + c;
}

/// Given triangle ABC and ball radius, finds two centers of balls each touching all 3 triangle's vertices;
/// \return false if such balls do not exist (radius is smaller that circumcircle radius)
template <typename T>
[[nodiscard]] bool circumballCenters( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c, T radius,
    Vector3<T> & centerPos, // ball's center from the positive side of triangle
    Vector3<T> & centerNeg )// ball's center from the negative side of triangle
{
    const auto rr = sqr( radius );
    const auto circRadSq = circumcircleDiameterSq( a, b, c ) / T( 4 );
    if ( rr < circRadSq )
        return false;

    const auto x = std::sqrt( rr - circRadSq );
    const auto xn = x * normal( a, b, c );
    const auto circCenter = circumcircleCenter( a, b, c );
    centerPos = circCenter + xn;
    centerNeg = circCenter - xn;

    return true;
}

/// Computes sine of minimal angle in ABC triangle, which is equal to ratio of minimal edge length to circumcircle diameter
/// \ingroup MathGroup
template <typename T>
[[nodiscard]] T minTriangleAngleSin( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c )
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
[[nodiscard]] T minTriangleAngle( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c )
{
    return std::asin( minTriangleAngleSin( a, b, c ) );
}

/// Aspect ratio of a triangle is the ratio of the circum-radius to twice its in-radius
/// \ingroup MathGroup
template<typename T>
[[nodiscard]] T triangleAspectRatio( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c )
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

/// computes directed double area of given triangle
template<typename T>
[[nodiscard]] inline Vector3<T> dirDblArea( const Triangle3<T> & t )
{
    return cross( t[1] - t[0], t[2] - t[0] );
}

/// computes directed double area of given triangle
template<typename T>
[[nodiscard]] inline Vector3<T> dirDblArea( const Vector3<T> & p, const Vector3<T> & q, const Vector3<T> & r )
{
    return cross( q - p, r - p );
}

/// computes unitNormal of given triangle
template<typename T>
[[nodiscard]] inline Vector3<T> normal( const Vector3<T> & p, const Vector3<T> & q, const Vector3<T> & r )
{
    return dirDblArea( p, q, r ).normalized();
}

/// computes the square of double area of given triangle
template<typename T>
[[nodiscard]] inline T dblAreaSq( const Vector3<T> & p, const Vector3<T> & q, const Vector3<T> & r )
{
    return dirDblArea( p, q, r ).lengthSq();
}

/// computes twice the area of given triangle
template<typename T>
[[nodiscard]] inline T dblArea( const Vector3<T> & p, const Vector3<T> & q, const Vector3<T> & r )
{
    return dirDblArea( p, q, r ).length();
}

/// computes twice the area of given triangle
template<typename T>
[[nodiscard]] inline T area( const Vector3<T> & p, const Vector3<T> & q, const Vector3<T> & r )
{
    return dblArea( p, q, r ) / 2; 
}

/// computes twice the area of given triangle
template<typename T>
[[nodiscard]] inline T dblArea( const Vector2<T> & p, const Vector2<T> & q, const Vector2<T> & r )
{
    return std::abs( cross( q - p, r - p ) );
}

/// computes twice the area of given triangle
template<typename T>
[[nodiscard]] inline T area( const Vector2<T> & p, const Vector2<T> & q, const Vector2<T> & r )
{
    return dblArea( p, q, r ) / 2;
}

/// make degenerate triangle (all 3 points on a line) that maximally resembles the input one and has the same centroid
template <typename T>
[[nodiscard]] Triangle3<T> makeDegenerate( const Triangle3<T> & t )
{
    const auto c = ( t[0] + t[1] + t[2] ) / T(3);
    int longest = 0;
    T longestSq = 0;
    for ( int i = 0; i < 3; ++i )
    {
        const auto sq = ( t[i] - c ).lengthSq();
        if ( longestSq >= sq )
            continue;
        longest = i;
        longestSq = sq;
    }
    const auto d = ( t[longest] - c ).normalized();

    // project triangle on the line (c, d)
    Triangle3<T> res;
    for ( int i = 0; i < 3; ++i )
        res[i] = c + d * dot( d, t[i] - c );
    return res;
}

/// project given triangle on a plane passing via its centroid and having unit normal (n);
/// if after projection triangle normal turns out to be inversed, then collapses the triangle into degenerate line segment
template <typename T>
[[nodiscard]] Triangle3<T> triangleWithNormal( const Triangle3<T> & t, const Vector3<T> & n )
{
    const auto c = ( t[0] + t[1] + t[2] ) / T(3);
    Triangle3<T> res;
    for ( int i = 0; i < 3; ++i )
        res[i] = t[i] - n * dot( n, t[i] - c );

    if ( dot( n, dirDblArea( res ) ) < 0 ) // projected triangle has inversed normal
        res = makeDegenerate( res );
    return res;
}

/// given an edge direction between two faces with given normals, computes sine of dihedral angle between the faces:
/// 0 if both faces are in the same plane,
/// positive if the faces form convex surface,
/// negative if the faces form concave surface
/// \ingroup MathGroup
template <typename T>
[[nodiscard]] T dihedralAngleSin( const Vector3<T>& leftNorm, const Vector3<T>& rightNorm, const Vector3<T>& edgeVec )
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
[[nodiscard]] T dihedralAngleCos( const Vector3<T>& leftNorm, const Vector3<T>& rightNorm )
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
[[nodiscard]] T dihedralAngle( const Vector3<T>& leftNorm, const Vector3<T>& rightNorm, const Vector3<T>& edgeVec )
{
    auto sin = dihedralAngleSin( leftNorm, rightNorm, edgeVec );
    auto cos = dihedralAngleCos( leftNorm, rightNorm );
    return std::atan2( sin, cos );
}

/// given the lengths of 3 edges of triangle ABC, and
/// assuming that point B has coordinates (0,0); point A - (0,c);
/// computes the coordinates of point C (where c.x >= 0) or returns std::nullopt if input lengths are invalid for a triangle
template <typename T>
[[nodiscard]] std::optional<Vector2<T>> posFromTriEdgeLengths( T a, T b, T c )
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

/// given the lengths of 4 edges of a quadrangle, and one of its diagonals (c);
/// returns the length of the other diagonal if the quadrangle is valid and convex or std::nullopt otherwise
template <typename T>
[[nodiscard]] std::optional<T> quadrangleOtherDiagonal( T a, T b, T c, T a1, T b1 )
{
    const auto p = posFromTriEdgeLengths( a, b, c );
    if ( !p )
        return {};
    auto p1 = posFromTriEdgeLengths( a1, b1, c );
    if ( !p1 )
        return {};
    p1->x = -p1->x;
    //where the other diagonal crosses axis Oy
    auto y = ( p->x * p1->y - p1->x * p->y ) / ( p->x - p1->x );
    if ( y < 0 || y > c )
        return {};
    return ( *p - *p1 ).length();
}

} // namespace MR
