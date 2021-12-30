#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// >0 for clockwise loop, < 0 for CCW loop
template<typename T>
T calcOrientedArea( const Contour2<T> & contour )
{
    if ( contour.size() < 3 )
        return 0;

    T area = 0;
    auto p0 = contour[0];

    for ( int i = 2; i < contour.size(); ++i )
    {
        auto p1 = contour[i - 1];
        auto p2 = contour[i];
        area += cross( p2 - p0, p1 - p0 );
    }

    return area / 2;
}

// copy double-contour to float-contour, or vice versa
template<typename To, typename From>
To copyContour( const From & from )
{
    To res;
    res.reserve( from.size() );
    for ( const auto & p : from )
        res.emplace_back( p );
    return res;
}

// copy double-contours to float-contours, or vice versa
template<typename To, typename From>
To copyContours( const From & from )
{
    To res;
    res.reserve( from.size() );
    for ( const auto & c : from )
        res.push_back( copyContour<typename To::value_type>( c ) );
    return res;
}

} //namespace MR
