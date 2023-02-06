#pragma once

#include "MRPlane3.h"
#include "MRLine3.h"
#include "MRLineSegm.h"
#include "MRVector2.h"
#include <optional>

namespace MR
{

/// \defgroup IntersectionGroup Intersection
/// \ingroup MathGroup
/// \{

/// finds an intersection between a plane1 and a plane2
/// \param plane1,plane2 should be normalized for check parallelism
/// \return nullopt if they are parallel (even if they match)
template<typename T>
std::optional<Line3<T>> intersection( const Plane3<T>& plane1, const Plane3<T>& plane2,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    const auto crossDir = cross( plane1.n, plane2.n );

    if ( crossDir.lengthSq() < errorLimit * errorLimit )
        return {};

    Matrix3<T> matrix( plane1.n, plane2.n, crossDir );
    const auto point = matrix.inverse() * Vector3<T>( plane1.d, plane2.d, 0 );

    return Line3<T>( point, crossDir.normalized() );
}

/// finds an intersection between a plane and a line
/// \param plane,line should be normalized for check parallelism
/// \return nullopt if they are parallel (even line belongs to plane)
template<typename T>
std::optional<Vector3<T>> intersection( const Plane3<T>& plane, const Line3<T>& line,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    const auto den = dot( plane.n, line.d );
    if ( std::abs(den) < errorLimit )
        return {};
    return line.p + ( plane.d - dot( plane.n, line.p ) ) / den * line.d;
}

/// finds an intersection between a line1 and a line2
/// \param line1,line2 should be normalized for check parallelism
/// \return nullopt if they are not intersect (even if they match)
template<typename T>
std::optional<Vector3<T>> intersection( const Line3<T>& line1, const Line3<T>& line2,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    const auto crossDir = cross( line1.d, line2.d );
    if ( crossDir.lengthSq() < errorLimit * errorLimit )
        return {};

    const auto p1 = dot( crossDir, line1.p );
    const auto p2 = dot( crossDir, line2.p );
    if ( std::abs( p1 - p2 ) >= errorLimit )
        return {};

    const auto n2 = cross( line2.d, crossDir );
    const T den = dot( line1.d, n2 );
    if ( den == 0 ) // check for calculation
        return {};
    return line1.p + dot( ( line2.p - line1.p ), n2 ) / den * line1.d;
}

/// finds an intersection between a segm1 and a segm2
/// \return nullopt if they don't intersect (even if they match)
inline std::optional<Vector2f> intersection( const LineSegm2f& segm1, const LineSegm2f& segm2 )
{
    auto avec = segm1.b - segm1.a;
    if ( cross( avec, segm2.a - segm1.a ) * cross( segm2.b - segm1.a, avec ) <= 0 )
        return {};
    auto bvec = segm2.b - segm2.a;
    auto cda = cross( bvec, segm1.a - segm2.a );
    auto cbd = cross( segm1.b - segm2.a, bvec );
    if ( cda * cbd <= 0 )
        return {};
    return ( segm1.b * cda + segm1.a * cbd ) / ( cda + cbd );
}

/// finds distance between a plane1 and a plane2
/// \return nullopt if they intersect
template<typename T>
std::optional<T> distance( const Plane3<T>& plane1, const Plane3<T>& plane2,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    const auto crossDir = cross( plane1.n, plane2.n );

    if ( crossDir.lengthSq() >= errorLimit * errorLimit )
        return {};

    return ( plane2.n * plane2.d - plane1.n * plane1.d ).length();
}

/// finds distance between a plane and a line;
/// \return nullopt if they intersect
template<typename T>
std::optional<T> distance( const Plane3<T>& plane, const Line3<T>& line,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    const auto den = dot( plane.n, line.d );
    if ( std::abs( den ) >= errorLimit )
        return {};

    return std::abs( dot( line.p, plane.n ) - plane.d );
}

/// finds the closest points between two lines in 3D;
/// for parallel lines the selection is arbitrary;
/// \return two equal points if the lines intersect
template<typename T>
LineSegm3<T> closestPoints( const Line3<T>& line1, const Line3<T>& line2 )
{
    const auto d11 = line1.d.lengthSq();
    const auto d12 = dot( line1.d, line2.d );
    const auto d22 = line2.d.lengthSq();
    const auto det = d12 * d12 - d11 * d22;
    if ( det == 0 )
    {
        // lines are parallel
        return { line1.p, line2.project( line1.p ) };
    }

    const auto dp = line2.p - line1.p;
    const auto x = dot( dp, line1.d ) / det;
    const auto y = dot( dp, line2.d ) / det;
    const auto a = d12 * y - d22 * x;
    const auto b = d11 * y - d12 * x;
    return { line1( a ), line2( b ) };
}

/// finds distance between parallel or skew lines ( a line1 and a line2 )
/// \return zero if they intersect
template<typename T>
T distance( const Line3<T>& line1, const Line3<T>& line2 )
{
    const auto cl = closestPoints( line1, line2 );
    return ( cl.a - cl.b ).length();
}

/// \}

} // namespace MR
