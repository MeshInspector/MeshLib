#pragma once

#include "MRPch/MRBindingMacros.h"
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

/// returns sum length of the given contour
/// \tparam R is the type for the accumulation and for result
template<typename V, typename R = typename V::ValueType>
R calcLength( const Contour<V>& contour )
{
    R l = R( 0 );
    for ( int i = 1; i < contour.size(); ++i )
        l += R( ( contour[i] - contour[i - 1] ).length() );
    return l;
}

/// Copy double-contour to float-contour, or vice versa. Also handles 2D-3D conversions (zeroing or dropping the Z component).
/// This is excluded from the bindings for simplicity. While it does bind (if manually instantiated),
///   the resulting names are quite ugly. Instead we provide wrapper functions with nicer names below.
template<typename To, typename From>
MR_BIND_IGNORE To convertContour( const From & from )
{
    To res;
    res.reserve( from.size() );
    for ( const auto & p : from )
        res.emplace_back( p );
    return res;
}

/// Copy double-contours to float-contours, or vice versa. Also handles 2D-3D conversions (zeroing or dropping the Z component).
/// This is excluded from the bindings for simplicity. While it does bind (if manually instantiated),
///   the resulting names are quite ugly. Instead we provide wrapper functions with nicer names below.
template<typename To, typename From>
MR_BIND_IGNORE To convertContours( const From & from )
{
    To res;
    res.reserve( from.size() );
    for ( const auto & c : from )
        res.push_back( convertContour<typename To::value_type>( c ) );
    return res;
}

/// \}

// Instantiate the templates when generating bindings.
MR_BIND_TEMPLATE( float calcOrientedArea( const Contour2<float> & contour ) )
MR_BIND_TEMPLATE( double calcOrientedArea( const Contour2<double> & contour ) )
MR_BIND_TEMPLATE( float calcLength( const Contour2<float>& contour ) )
MR_BIND_TEMPLATE( double calcLength( const Contour2<double>& contour ) )
MR_BIND_TEMPLATE( float calcLength( const Contour3<float>& contour ) )
MR_BIND_TEMPLATE( double calcLength( const Contour3<double>& contour ) )
MR_BIND_TEMPLATE( Vector3<float> calcOrientedArea( const Contour3<float> & contour ) )
MR_BIND_TEMPLATE( Vector3<double> calcOrientedArea( const Contour3<double> & contour ) )


template <typename From> [[nodiscard]] Contour2f convertContourTo2f( const From &from ) { return convertContour<Contour2f>( from ); }
template <typename From> [[nodiscard]] Contour3f convertContourTo3f( const From &from ) { return convertContour<Contour3f>( from ); }
template <typename From> [[nodiscard]] Contour2d convertContourTo2d( const From &from ) { return convertContour<Contour2d>( from ); }
template <typename From> [[nodiscard]] Contour3d convertContourTo3d( const From &from ) { return convertContour<Contour3d>( from ); }

template <typename From> [[nodiscard]] Contours2f convertContoursTo2f( const From &from ) { return convertContours<Contours2f>( from ); }
template <typename From> [[nodiscard]] Contours3f convertContoursTo3f( const From &from ) { return convertContours<Contours3f>( from ); }
template <typename From> [[nodiscard]] Contours2d convertContoursTo2d( const From &from ) { return convertContours<Contours2d>( from ); }
template <typename From> [[nodiscard]] Contours3d convertContoursTo3d( const From &from ) { return convertContours<Contours3d>( from ); }

MR_BIND_TEMPLATE( Contour2f convertContourTo2f( const Contour2f & from ) )
MR_BIND_TEMPLATE( Contour2f convertContourTo2f( const Contour2d & from ) )
MR_BIND_TEMPLATE( Contour2f convertContourTo2f( const Contour3f & from ) )
MR_BIND_TEMPLATE( Contour2f convertContourTo2f( const Contour3d & from ) )
MR_BIND_TEMPLATE( Contour2d convertContourTo2d( const Contour2f & from ) )
MR_BIND_TEMPLATE( Contour2d convertContourTo2d( const Contour2d & from ) )
MR_BIND_TEMPLATE( Contour2d convertContourTo2d( const Contour3f & from ) )
MR_BIND_TEMPLATE( Contour2d convertContourTo2d( const Contour3d & from ) )
MR_BIND_TEMPLATE( Contour3f convertContourTo3f( const Contour2f & from ) )
MR_BIND_TEMPLATE( Contour3f convertContourTo3f( const Contour2d & from ) )
MR_BIND_TEMPLATE( Contour3f convertContourTo3f( const Contour3f & from ) )
MR_BIND_TEMPLATE( Contour3f convertContourTo3f( const Contour3d & from ) )
MR_BIND_TEMPLATE( Contour3d convertContourTo3d( const Contour2f & from ) )
MR_BIND_TEMPLATE( Contour3d convertContourTo3d( const Contour2d & from ) )
MR_BIND_TEMPLATE( Contour3d convertContourTo3d( const Contour3f & from ) )
MR_BIND_TEMPLATE( Contour3d convertContourTo3d( const Contour3d & from ) )

MR_BIND_TEMPLATE( Contours2f convertContoursTo2f( const Contours2f & from ) )
MR_BIND_TEMPLATE( Contours2f convertContoursTo2f( const Contours2d & from ) )
MR_BIND_TEMPLATE( Contours2f convertContoursTo2f( const Contours3f & from ) )
MR_BIND_TEMPLATE( Contours2f convertContoursTo2f( const Contours3d & from ) )
MR_BIND_TEMPLATE( Contours2d convertContoursTo2d( const Contours2f & from ) )
MR_BIND_TEMPLATE( Contours2d convertContoursTo2d( const Contours2d & from ) )
MR_BIND_TEMPLATE( Contours2d convertContoursTo2d( const Contours3f & from ) )
MR_BIND_TEMPLATE( Contours2d convertContoursTo2d( const Contours3d & from ) )
MR_BIND_TEMPLATE( Contours3f convertContoursTo3f( const Contours2f & from ) )
MR_BIND_TEMPLATE( Contours3f convertContoursTo3f( const Contours2d & from ) )
MR_BIND_TEMPLATE( Contours3f convertContoursTo3f( const Contours3f & from ) )
MR_BIND_TEMPLATE( Contours3f convertContoursTo3f( const Contours3d & from ) )
MR_BIND_TEMPLATE( Contours3d convertContoursTo3d( const Contours2f & from ) )
MR_BIND_TEMPLATE( Contours3d convertContoursTo3d( const Contours2d & from ) )
MR_BIND_TEMPLATE( Contours3d convertContoursTo3d( const Contours3f & from ) )
MR_BIND_TEMPLATE( Contours3d convertContoursTo3d( const Contours3d & from ) )

} // namespace MR
