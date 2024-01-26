#pragma once

#include "MRVector3.h"
#include "MRSegmPoint.h"

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
    T a; ///< a in [0,1], a=0 => point is on [v2,v0] edge, a=1 => point is in v1
    T b; ///< b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2

    static constexpr auto eps = SegmPoint<T>::eps;

    constexpr TriPoint() noexcept : a( 0 ), b( 0 ) { }
    explicit TriPoint( NoInit ) noexcept { }
    constexpr TriPoint( T a, T b ) noexcept : a( a ), b( b ) { }
    template <typename U>
    constexpr TriPoint( const TriPoint<U> & s ) : a( T( s.a ) ), b( T( s.b ) ) { }

    /// given a point coordinates and triangle (v0,v1,v2) computes barycentric coordinates of the point
    TriPoint( const Vector3<T> & p, const Vector3<T> & v0, const Vector3<T> & v1, const Vector3<T> & v2 ) : TriPoint( p - v0, v1 - v0, v2 - v0 ) { }
    /// given a point coordinates and triangle (0,v1,v2) computes barycentric coordinates of the point
    TriPoint( const Vector3<T> & p, const Vector3<T> & v1, const Vector3<T> & v2 );

    /// given three values in three vertices, computes interpolated value at this barycentric coordinates
    template <typename U>
    U interpolate( const U & v0, const U & v1, const U & v2 ) const
        { return ( 1 - a - b ) * v0 + a * v1 + b * v2; }

    /// represents the same point relative to next edge in the same triangle
    [[nodiscard]] TriPoint lnext() const { return { b, 1 - a - b }; }

    // requirements:
    // 1) inVertex() == onEdge() && toEdge()->inVertex()
    // 2) invariance to lnext() application

    /// returns [0,2] if the point is in a vertex or -1 otherwise
    constexpr int inVertex() const;
    /// returns [0,2] if the point is on edge or -1 otherwise:
    /// 0 means edge [v1,v2]; 1 means edge [v2,v0]; 2 means edge [v0,v1]
    constexpr int onEdge() const;

    /// returns true if two points have equal (a,b) representation
    [[nodiscard]] constexpr bool operator==( const TriPoint& rhs ) const = default;
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
constexpr int TriPoint<T>::inVertex() const
{
    if ( a <= eps && b <= eps )
        return 0;
    if ( 1 - a - b <= eps )
    {
        if ( b <= eps )
            return 1;
        if ( a <= eps )
            return 2;
    }
    return -1;
}

template <typename T>
constexpr int TriPoint<T>::onEdge() const
{
    if ( 1 - a - b <= eps )
        return 0;
    if ( a <= eps )
        return 1;
    if ( b <= eps )
        return 2;
    return -1;
}

/// \}

} // namespace MR
