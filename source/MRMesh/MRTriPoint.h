#pragma once

#include "MRVector3.h"
#include "MRSegmPoint.h"
#include "MRMesh/MRMacros.h"
#include <iosfwd>

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
    template <typename U> MR_REQUIRES_IF_SUPPORTED(!std::is_same_v<T, U>) // The condition is needed for the bindings to not duplicate this ctor against the copy ctor.
    constexpr TriPoint( const TriPoint<U> & s ) : a( T( s.a ) ), b( T( s.b ) ) { }

    /// given a point coordinates and triangle (v0,v1,v2) computes barycentric coordinates of the point;
    /// if the point is known to be on an edge, numbered as in onEdge(), then pass it to place the point exactly there
    TriPoint( const Vector3<T> & p, const Vector3<T> & v0, const Vector3<T> & v1, const Vector3<T> & v2, int edge = -1 ) : TriPoint( p - v0, v1 - v0, v2 - v0, edge ) { }
    /// given a point coordinates and triangle (0,v1,v2) computes barycentric coordinates of the point;
    /// if the point is known to be on an edge, numbered as in onEdge(), then pass it to place the point exactly there
    TriPoint( const Vector3<T> & p, const Vector3<T> & v1, const Vector3<T> & v2, int edge = -1 );

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

    friend std::ostream& operator<<( std::ostream& s, const TriPoint& tp )
    {
        return s << tp.a << ' ' << tp.b;
    }

    friend std::istream& operator>>( std::istream& s, TriPoint& tp )
    {
        s >> tp.a >> tp.b;
        return s;
    }
};

/// \related TriPoint
/// \{

template <typename T>
TriPoint<T>::TriPoint( const Vector3<T> & p, const Vector3<T> & v1, const Vector3<T> & v2, int edge )
{
    if ( edge >= 0 )
    {
        // the edge opposite to vertex #edge
        const Vector3<T> v0;
        const Vector3<T> * v[3] = { &v0, &v1, &v2 };
        const auto & org = *v[( edge + 1 ) % 3];
        const auto d = *v[( edge + 2 ) % 3] - org;
        const T dSq = d.lengthSq();
        const T s = dSq > 0 ? std::clamp( dot( p - org, d ) / dSq, T(0), T(1) ) : T(0);
        switch ( edge )
        {
        case 0:
            a = 1 - s;
            b = s;
            break;
        case 1:
            a = 0;
            b = 1 - s;
            break;
        default:
            a = s;
            b = 0;
        }
        return;
    }
    // by the areas of the triangles the point makes with the edges, which keeps the precision in thin triangles
    const auto n = cross( v1, v2 );
    const T nSq = n.lengthSq();
    if ( nSq <= 0 )
    {
        // degenerate triangle
        a = b = 1 / T(3);
        return;
    }
    const T inv = 1 / nSq;
    a = std::clamp( inv * dot( n, cross( p, v2 ) ), T(0), T(1) );
    b = std::clamp( inv * dot( n, cross( v1, p ) ), T(0), T(1) - a );
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
