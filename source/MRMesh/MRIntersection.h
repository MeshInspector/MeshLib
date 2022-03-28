#pragma once

#include "MRPlane3.h"
#include "MRLine3.h"
#include "MRLineSegm.h"
#include <optional>

namespace MR
{

template<typename T>
const T errorLimit = std::numeric_limits<T>::epsilon() * 20.f;

// finds an intersection between a plane1 and a plane2
// returns nullopt if they are parallel
// returns line from plane1 if plane1 equal plane2
template<typename T>
std::optional<Line3<T>> intersection( const Plane3<T>& plane1, const Plane3<T>& plane2 )
{
    const auto plane1N = plane1.normalized();
    const auto plane2N = plane2.normalized();
    const auto crossDir = cross( plane1N.n, plane2N.n );

    const auto planeNpoint = plane1N.n * plane1N.d;
    const auto plane2Npoint = plane2N.n * plane2N.d;

    if ( crossDir.length() < errorLimit<T> )
    {
        if ( ( planeNpoint - plane2Npoint ).length() < errorLimit<T> )
        {
            auto bv = plane1N.n.furthestBasisVector();
            return Line3<T>( planeNpoint, cross( plane1N.n, bv ).normalized() );
        }
        else
            return {};
    }

    const auto p1 = plane1N.project( plane2Npoint );
    const auto p2 = plane2N.project( planeNpoint );
    const auto p3 = Line3<T>(plane2Npoint, ( p1 - plane2Npoint ) ).project(p2);
    return Line3<T>( plane2Npoint + ( p2 - plane2Npoint ) * ( p1 - plane2Npoint ).length() / ( p3 - plane2Npoint ).length(), crossDir.normalized() );
}

// finds an intersection between a plane and a line;
// returns nullopt if they are parallel
template<typename T>
std::optional<Vector3<T>> intersection( const Plane3<T> & plane, const Line3<T> & line )
{
    const auto planeN = plane.normalized();
    const auto lineN = line.normalized();

    const auto den = dot( planeN.n, lineN.d );
    if ( den == 0 )
        return {};
    return line.p + ( plane.d - dot( plane.n, line.p ) ) / den * line.d;
}

// finds an intersection between a line1 and a line2
// returns nullopt if they are not intersect
template<typename T>
std::optional<Vector3<T>> intersection( const Line3<T>& line1, const Line3<T>& line2 )
{
    const auto line1N = line1.normalized();
    const auto line2N = line2.normalized();
    const auto crossDir = cross( line1N.d, line2N.d );
    if ( crossDir.length() < errorLimit<T> )
    {
        if ( ( line1N.project( line2N.p ) - line2N.p ).length() < errorLimit<T> )
            return line1N.p;
        else
            return {};
    }
    
    const auto p1 = dot( crossDir, line1N.p );
    const auto p2 = dot( crossDir, line2N.p );
    if ( ( p1 - p2 ) >= errorLimit<T> )
        return {};

    const auto n2 = cross( line2N.d, crossDir );
    return line1N.p + dot( ( line2N.p - line1N.p ), n2 ) / dot( line1N.d, n2 ) * line1N.d;
}




// finds closest point on a plane1 and a plane2;
// returns nullopt if they are intersect
template<typename T>
std::optional<LineSegm3<T>> closestPoints( const Plane3<T>& plane1, const Plane3<T>& plane2 )
{
    const auto plane1N = plane1.normalized();
    const auto plane2N = plane2.normalized();
    const auto crossDir = cross( plane1N.n, plane2N.n );

    if ( crossDir.length() >= errorLimit<T> )
        return {};

    const auto planeNpoint = plane1N.n * plane1N.d;
    const auto plane2Npoint = plane2N.n * plane2N.d;
    return LineSegm3( planeNpoint, plane2Npoint );
}

// finds closest point on a plane and a line;
// returns nullopt if they are intersect
template<typename T>
std::optional<LineSegm3<T>> closestPoints( const Plane3<T>& plane, const Line3<T>& line )
{
    const auto planeN = plane.normalized();
    const auto lineN = line.normalized();

    const auto den = dot( planeN.n, lineN.d );
    if ( std::abs(den) >= errorLimit<T> )
        return {};

    return LineSegm3( planeN.project( lineN.p ), lineN.p );
}

// finds closest point on a line1 and a line2;
// returns nullopt if they are intersect
template<typename T>
std::optional<LineSegm3<T>> closestPoints( const Line3<T>& line1, const Line3<T>& line2 )
{
    const auto line1N = line1.normalized();
    const auto line2N = line2.normalized();
    const auto crossDir = cross( line1N.d, line2N.d );
    if ( crossDir.length() < errorLimit<T> )
        return LineSegm3( line1N.p, line2N.project( line1N.p ) );

    const auto p1 = dot( crossDir, line1N.p );
    const auto p2 = dot( crossDir, line2N.p );
    if ( ( p1 - p2 ) < errorLimit<T> )
        return {};

    const auto n2 = cross( line2N.d, crossDir );
    const auto closest1 = line1N.p + dot( ( line2N.p - line1N.p ), n2 ) / dot( line1N.d, n2 ) * line1N.d;
    const auto closest2 = closest1 + ( p2 - p1 ) * crossDir;
    return LineSegm3( closest1, closest2 );
}

} //namespace MR
