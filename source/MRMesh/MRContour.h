#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

namespace MR
{

/// \defgroup ContourGroup Contour
/// \ingroup MathGroup
/// \{

/// >0 for clockwise loop, < 0 for CCW loop
/// \tparam R is the type for the accumulation and for result
template<typename T, typename R = T>
R calcOrientedArea( const Contour2<T> & contour )
{
    if ( contour.size() < 3 )
        return 0;

    R area = 0;
    Vector2<R> p0{ contour[0] };

    for ( int i = 2; i < contour.size(); ++i )
    {
        Vector2<R> p1{ contour[i - 1] };
        Vector2<R> p2{ contour[i] };
        area += cross( p2 - p0, p1 - p0 );
    }

    return R(0.5) * area;
}

/// returns the vector with the magnitude equal to contour area, and directed to see the contour
/// in ccw order from the vector tip
/// \tparam R is the type for the accumulation and for result
template<typename T, typename R = T>
Vector3<R> calcOrientedArea( const Contour3<T> & contour )
{
    if ( contour.size() < 3 )
        return {};

    Vector3<R> area;
    Vector3<R> p0{ contour[0] };

    for ( int i = 2; i < contour.size(); ++i )
    {
        Vector3<R> p1{ contour[i - 1] };
        Vector3<R> p2{ contour[i] };
        area += cross( p1 - p0, p2 - p0 );
    }

    return R(0.5) * area;
}

/// copy double-contour to float-contour, or vice versa
template<typename To, typename From>
To copyContour( const From & from )
{
    To res;
    res.reserve( from.size() );
    for ( const auto & p : from )
        res.emplace_back( p );
    return res;
}

/// copy double-contours to float-contours, or vice versa
template<typename To, typename From>
To copyContours( const From & from )
{
    To res;
    res.reserve( from.size() );
    for ( const auto & c : from )
        res.push_back( copyContour<typename To::value_type>( c ) );
    return res;
}

/// \}

} // namespace MR
