#pragma once

#include "MRPlane3.h"
#include "MRLine3.h"
#include "MRLineSegm.h"
#include <optional>

namespace MR
{

template<typename T>
const T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 );

// finds an intersection between a plane1 and a plane2
// plane1 and plane2 should be normalized for check parallelism
// returns nullopt if they are parallel (even if they match)
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

// finds an intersection between a plane and a line
// plane and line should be normalized for check parallelism
// returns nullopt if they are parallel (even line belongs to plane)
template<typename T>
std::optional<Vector3<T>> intersection( const Plane3<T>& plane, const Line3<T>& line,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    const auto den = dot( plane.n, line.d );
    if ( std::abs(den) < errorLimit )
        return {};
    return line.p + ( plane.d - dot( plane.n, line.p ) ) / den * line.d;
}

// finds an intersection between a line1 and a line2
// plane and line should be normalized for check parallelism
// returns nullopt if they are not intersect (even if they match)
template<typename T>
std::optional<Vector3<T>> intersection( const Line3<T>& line1, const Line3<T>& line2,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    const auto crossDir = cross( line1.d, line2.d );
    if ( crossDir.lengthSq() < errorLimit * errorLimit )
        return {};

    const auto p1 = dot( crossDir, line1.p );
    const auto p2 = dot( crossDir, line2.p );
    if ( ( p1 - p2 ) >= errorLimit )
        return {};

    const auto n2 = cross( line2.d, crossDir );
    return line1.p + dot( ( line2.p - line1.p ), n2 ) / dot( line1.d, n2 ) * line1.d;
}




// finds closest point on a plane1 and a plane2;
// returns nullopt if they intersect
template<typename T>
std::optional<LineSegm3<T>> closestPoints( const Plane3<T>& plane1, const Plane3<T>& plane2,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    const auto crossDir = cross( plane1.n, plane2.n );

    if ( crossDir.lengthSq() >= errorLimit * errorLimit )
        return {};

    const auto plane1Point = plane1.n * plane1.d;
    const auto plane2Point = plane2.n * plane2.d;
    return LineSegm3<T>( plane1Point, plane2Point );
}

// finds closest point on a plane and a line;
// returns nullopt if they intersect
template<typename T>
std::optional<LineSegm3<T>> closestPoints( const Plane3<T>& plane, const Line3<T>& line,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    const auto den = dot( plane.n, line.d );
    if ( std::abs( den ) >= errorLimit )
        return {};

    return LineSegm3<T>( plane.project( line.p ), line.p );
}

// finds closest point on a line1 and a line2;
// returns nullopt if they intersect
template<typename T>
std::optional<LineSegm3<T>> closestPoints( const Line3<T>& line1, const Line3<T>& line2,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    const auto crossDir = cross( line1.d, line2.d );
    if ( crossDir.lengthSq() < errorLimit * errorLimit )
        return LineSegm3<T>( line1.project( line2.p ), line2.p );

    const auto p1 = dot( crossDir, line1.p );
    const auto p2 = dot( crossDir, line2.p );
    if ( ( p1 - p2 ) < errorLimit )
        return {};

    const auto n2 = cross( line2.d, crossDir );
    const auto closest1 = line1.p + dot( ( line2.p - line1.p ), n2 ) / dot( line1.d, n2 ) * line1.d;
    const auto closest2 = closest1 + ( p2 - p1 ) * crossDir;
    return LineSegm3<T>( closest1, closest2 );
}

} //namespace MR
