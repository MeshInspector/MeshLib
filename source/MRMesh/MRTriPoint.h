#pragma once

#include "MRVector3.h"
#include <limits>

namespace MR
{

/// \brief encodes a point inside a triangle using barycentric coordinates
/// \ingroup MathGroup
/// \details Notations used below: v0, v1, v2 - points of the triangle
template <typename T>
struct TriPoint
{

    /// barycentric coordinates:
    /// a+b in [0,1], a+b=0 => point is in v0, a+b=1 => point is on [v1,v2] edge
    T a = 0; ///< a in [0,1], a=0 => point is on [v2,v0] edge, a=1 => point is in v1
    T b = 0; ///< b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2

    static constexpr auto eps = 10 * std::numeric_limits<T>::epsilon();

    TriPoint() = default;
    TriPoint( T a, T b ) : a( a ), b( b ) { }

    /// given a point coordinates and triangle (v0,v1,v2) computes barycentric coordinates of the point
    TriPoint( const Vector3<T> & p, const Vector3<T> & v0, const Vector3<T> & v1, const Vector3<T> & v2 ) : TriPoint( p - v0, v1 - v0, v2 - v0 ) { }
    /// given a point coordinates and triangle (0,v1,v2) computes barycentric coordinates of the point
    TriPoint( const Vector3<T> & p, const Vector3<T> & v1, const Vector3<T> & v2 );

    /// given three values in three vertices, computes interpolated value at this barycentric coordinates
    template <typename U>
    U interpolate( const U & v0, const U & v1, const U & v2 ) const
    {
        return ( 1 - a - b ) * v0 + a * v1 + b * v2;
    }

    /// returns [0,2] if the point is in a vertex or -1 otherwise
    int inVertex() const;
    /// returns [0,2] if the point is on edge or -1 otherwise:
    /// 0 means edge [v1,v2]; 1 means edge [v2,v0]; 2 means edge [v0,v1]
    int onEdge() const;
};

/// \related TriPoint
/// \{

template <typename T>
TriPoint<T>::TriPoint( const Vector3<T> & p, const Vector3<T> & v1, const Vector3<T> & v2 )
{
    const T v11 = dot( v1, v1 );
    const T v12 = dot( v1, v2 );
    const T v22 = dot( v2, v2 );
    const T det = v11 * v22 - v12 * v12;
    if ( det <= 0 )
    {
        // degenerate triangle
        a = b = 1 / T(3);
        return;
    }
    const T pv1 = dot( p, v1 );
    const T pv2 = dot( p, v2 );
    a = std::clamp( ( 1 / det ) * ( v22 * pv1 - v12 * pv2 ), T(0), T(1) );
    b = std::clamp( ( 1 / det ) * (-v12 * pv1 + v11 * pv2 ), T(0), T(1) - a );
}

template <typename T>
int TriPoint<T>::inVertex() const
{
    if ( a + b <= eps )
        return 0;
    if ( a + eps >= 1 )
        return 1;
    if ( b + eps >= 1 )
        return 2;
    return -1;
}

template <typename T>
int TriPoint<T>::onEdge() const
{
    // additional statements to guarantee:
    // MeshTriPoint.inVertex( topology ) == MeshTriPoint.onEdge() && MeshTriPoint.onEdge( topology )->inVertex( topology )

    if ( a + b + eps >= 1 )
    {
        if ( a + eps >= 1 )
            return 2; // mesh edge point will have 'a' rep
        return 0; // mesh edge point will have 'b' rep
    }

    if ( a <= eps )
    {
        if ( a + b <= eps )
            return 2; // mesh edge point will have 'a' rep
        return 1; // mesh edge point will have '1-b' rep
    }

    if ( b <= eps )
    {
        if ( a + b <= eps )
            return 1; // mesh edge point will have '1-b' rep
        return 2; // mesh edge point will have 'a' rep
    }

    return -1;
}

/// \}

} // namespace MR
