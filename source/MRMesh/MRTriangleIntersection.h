#pragma once

#include "MRVector3.h"
#include "MRIntersectionPrecomputes.h"
#include "MRTriPoint.h"
#include "MRTriMath.h"
#include "MRLineSegm.h"

#include <algorithm>
#include <optional>

namespace MR
{

/// \defgroup TriangleIntersectionGroup Triangle intersection
/// \ingroup MathGroup
/// \{

struct TriIntersectResult
{
    // barycentric representation
    TriPointf bary;
    // distance from ray origin to p in dir length units
    float t = 0;
    TriIntersectResult(float U, float V, float dist)
    {
        bary.a = U; bary.b = V; t = dist;
    }
};

/// given triangle ABC, rotates its vertices to make
/// segment AB the longest on exit
template <typename T>
void rotateToLongestEdge( Vector3<T>& a, Vector3<T>& b, Vector3<T>& c )
{
    const auto ab2 = distanceSq( a, b );
    const auto bc2 = distanceSq( b, c );
    const auto ca2 = distanceSq( c, a );
    if ( ab2 >= bc2 && ab2 >= ca2 )
        return;

    if ( bc2 >= ca2 )
    {
        assert( bc2 >= ab2 );
        auto t = a;
        a = b;
        b = c;
        c = t;
        return;
    }

    assert( ca2 >= ab2 );
    assert( ca2 >= bc2 );
    auto t = a;
    a = c;
    c = b;
    b = t;
}

/// checks whether triangles ABC and DEF intersect
/// returns false if ABC and DEF are coplanar;
/// due to floating-point errors inside, the result can be wrong in case of various degenerations of input triangles,
/// please consider using \ref findTriTriDistance function instead that is more tolerant to floating-point errors
template <typename T>
bool doTrianglesIntersect(
    Vector3<T> a, Vector3<T> b, Vector3<T> c,
    Vector3<T> d, Vector3<T> e, Vector3<T> f
)
{
    if ( dirDblArea( a, b, c ) == Vector3<T>{} ) // triangle ABC is degenerate
    {
        rotateToLongestEdge( a, b, c );
        return doTriangleSegmentIntersect( d, e, f, a, b );
    }
    if ( dirDblArea( d, e, f ) == Vector3<T>{} ) // triangle DEF is degenerate
    {
        rotateToLongestEdge( d, e, f );
        return doTriangleSegmentIntersect( a, b, c, d, e );
    }

    const auto abcd = mixed( a - d, b - d, c - d );
    const auto abce = mixed( a - e, b - e, c - e );
    const auto abcf = mixed( a - f, b - f, c - f );
    const auto abc_de = abcd * abce >= 0; // segment DE is located at one side of the plane ABC
    const auto abc_fd = abcf * abcd >= 0; // segment FD is located at one side of the plane ABC

    if ( abc_de && abc_fd && abce * abcf >= 0 )
        return false; // triangle DEF is located at one side of the plane ABC

    const auto defa = mixed( d - a, e - a, f - a );
    const auto defb = mixed( d - b, e - b, f - b );
    const auto defc = mixed( d - c, e - c, f - c );
    const auto def_ab = defa * defb >= 0;  // segment AB is located at one side of the plane DEF
    const auto def_ca = defc * defa >= 0;  // segment CA is located at one side of the plane DEF

    if ( def_ab && def_ca && defb * defc >= 0 )
        return false; // triangle ABC is located at one side of the plane DEF

    if ( abc_de )
        std::swap( d, f );
    else if( abc_fd )
        std::swap( d, e );
    // now segments DE and FD are crossed by the plane ABC: D at one side and EF at the other

    if ( def_ab )
        std::swap( a, c );
    else if ( def_ca )
        std::swap( a, b );
    // now segments AB and CA are crossed by the plane DEF: A at one side and BC at the other

    const auto abde = mixed( a - e, b - e, d - e );
    const auto abdf = mixed( a - f, b - f, d - f );

    if ( abde * abdf < 0 )
        return true; // AB segment penetrates triangle DEF since points E and F are at distinct sides of ABD

    const auto acde = mixed( a - e, c - e, d - e );

    if ( abde * acde < 0 )
        return true; // DE segment penetrates triangle ABC since points B and C are at distinct sides of ADE

    if ( abdf == 0 && acde == 0 )
        return true; // AB and DF segments are in the same plane, and AC and DE segments are in other same plane => triangles intersect, but no edge intersect the interior of other triangle

    const auto acdf = mixed( a - f, c - f, d - f );

    if ( acde * acdf < 0 )
        return true; // AC segment penetrates triangle DEF since points E and F are at distinct sides of ACD

    if ( abdf * acdf < 0 )
        return true; // DF segment penetrates triangle ABC since points B and C are at distinct sides of ADF

    if ( abde == 0 && acdf == 0 )
        return true; // AB and DE segments are in the same plane, and AC and DF segments are in other same plane => triangles intersect, but no edge intersect the interior of other triangle

    return false;
}

/// returns true if ABC plane contains point P
template<typename T>
bool isPointInPlane( const Vector3<T>& p, const Vector3<T>& a, const Vector3<T>& b, const Vector3<T>& c )
{
    return mixed( p - a, p - b, p - c ) == T( 0 );
}

MR_BIND_TEMPLATE( bool isPointInPlane( const Vector3<float>& p, const Vector3<float>& a, const Vector3<float>& b, const Vector3<float>& c ) )
MR_BIND_TEMPLATE( bool isPointInPlane( const Vector3<double>& p, const Vector3<double>& a, const Vector3<double>& b, const Vector3<double>& c ) )

/// returns true if AB line contains point P
template<typename T>
bool isPointInLine( const Vector3<T>& p, const Vector3<T>& a, const Vector3<T>& b )
{
    return cross( p - a, p - b ).lengthSq() == T( 0 );
}

/// returns true if AB line contains point P
template<typename T>
bool isPointInLine( const Vector2<T>& p, const Vector2<T>& a, const Vector2<T>& b )
{
    return cross( p - a, p - b ) == T( 0 );
}

/// returns true if AB segment contains point P
template<typename T>
bool isPointInSegm( const Vector3<T>& p, const Vector3<T>& a, const Vector3<T>& b )
{
    if ( !isPointInLine( p, a, b ) )
        return false;

    return dot( p - a, b - a ) >= 0 && dot( p - b, a - b ) >= 0;
}

/// returns true if AB segment contains point P
template<typename T>
bool isPointInSegm( const Vector2<T>& p, const Vector2<T>& a, const Vector2<T>& b )
{
    if ( !isPointInLine( p, a, b ) )
        return false;

    return dot( p - a, b - a ) >= 0 && dot( p - b, a - b ) >= 0;
}

MR_BIND_TEMPLATE( bool isPointInLine( const Vector3<float>& p, const Vector3<float>& a, const Vector3<float>& b ) )
MR_BIND_TEMPLATE( bool isPointInLine( const Vector3<double>& p, const Vector3<double>& a, const Vector3<double>& b ) )
MR_BIND_TEMPLATE( bool isPointInSegm( const Vector3<float>& p, const Vector3<float>& a, const Vector3<float>& b ) )
MR_BIND_TEMPLATE( bool isPointInSegm( const Vector3<double>& p, const Vector3<double>& a, const Vector3<double>& b ) )

/// returns true if ABC triangle contains point P
template<typename T>
bool isPointInTriangle( const Vector3<T>& p, const Vector3<T>& a, const Vector3<T>& b, const Vector3<T>& c )
{
    if ( !isPointInPlane( p, a, b, c ) )
        return false;
    const auto normDir = cross( b - a, c - a );
    if ( dot( normDir, cross( b - a, p - a ) ) < 0 )
        return false;
    if ( dot( normDir, cross( c - b, p - b ) ) < 0 )
        return false;
    if ( dot( normDir, cross( a - c, p - c ) ) < 0 )
        return false;
    if ( normDir.lengthSq() == 0 )
    {

        // ab parallel ac
        if ( a == b && b == c && p != a )
            return false; // fully degenerated
        if ( dot( b - a, c - a ) <= 0 )
            return isPointInSegm( p, b, c ); // ab ac looking in the opposite directions so check BC segm
        else if ( ( b - a ).lengthSq() > ( c - a ).lengthSq() )
            return isPointInSegm( p, a, b ); // ab ac looking in the same direction and AB is longer so check AB segm
        else
            return isPointInSegm( p, a, c ); // ab ac looking in the same direction and AC is longer so check AC segm
    }
    return true;
}

/// returns true if ABC triangle contains point P
template<typename T>
bool isPointInTriangle( const Vector2<T>& p, const Vector2<T>& a, const Vector2<T>& b, const Vector2<T>& c )
{
    const auto normSign = cross( b - a, c - a );
    if ( normSign * cross( b - a, p - a ) < 0 )
        return false;
    if ( normSign * cross( c - b, p - b ) < 0 )
        return false;
    if ( normSign * cross( a - c, p - c ) < 0 )
        return false;
    if ( normSign == 0 )
    {
        // ab parallel ac
        if ( a == b && b == c && p != a )
            return false; // fully degenerated
        if ( dot( b - a, c - a ) <= 0 )
            return isPointInSegm( p, b, c ); // ab ac looking in the opposite directions so check BC segm
        else if ( ( b - a ).lengthSq() > ( c - a ).lengthSq() )
            return isPointInSegm( p, a, b ); // ab ac looking in the same direction and AB is longer so check AB segm
        else
            return isPointInSegm( p, a, c ); // ab ac looking in the same direction and AC is longer so check AC segm
    }

    return true;
}

MR_BIND_TEMPLATE( bool isPointInTriangle( const Vector3<float>& p, const Vector3<float>& a, const Vector3<float>& b, const Vector3<float>& c ) )
MR_BIND_TEMPLATE( bool isPointInTriangle( const Vector3<double>& p, const Vector3<double>& a, const Vector3<double>& b, const Vector3<double>& c ) )
MR_BIND_TEMPLATE( bool isPointInTriangle( const Vector2<float>& p, const Vector2<float>& a, const Vector2<float>& b, const Vector2<float>& c ) )
MR_BIND_TEMPLATE( bool isPointInTriangle( const Vector2<double>& p, const Vector2<double>& a, const Vector2<double>& b, const Vector2<double>& c ) )

/// returns true if a plane containing edge XY separates point Z from triangle UVW
template <typename T>
bool doesEdgeXySeparate(
    const Vector3<T> & x, const Vector3<T> & y, const Vector3<T> & z,
    const Vector3<T> & u, const Vector3<T> & v, const Vector3<T> & w,
    Vector3<T> d // approximate normal of the plane
)
{
    const auto xy = ( y - x ).normalized();
    d = ( d - xy * dot( xy, d ) ).normalized();
    // now d is orthogonal to xy
    const auto dz = dot( d, z - x );
    return
        dz * dot( d, u - x ) < 0 &&
        dz * dot( d, v - x ) < 0 &&
        dz * dot( d, w - x ) < 0;
}

/// checks whether triangles ABC and DEF intersect;
/// it is designed to resolve false positives from \ref doTrianglesIntersect function
/// when two triangles are far apart but in one plane;
/// please consider using \ref findTriTriDistance function instead that is more tolerant to floating-point errors
template <typename T>
bool doTrianglesIntersectExt(
    const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c,
    const Vector3<T> & d, const Vector3<T> & e, const Vector3<T> & f
)
{
    if ( !doTrianglesIntersect( a, b, c, d, e, f ) )
        return false;

    // direction from center to center
    const auto dir = a + b + c - d - e - f;

    return
        !doesEdgeXySeparate( a, b, c, d, e, f, dir ) &&
        !doesEdgeXySeparate( b, c, a, d, e, f, dir ) &&
        !doesEdgeXySeparate( c, a, b, d, e, f, dir ) &&
        !doesEdgeXySeparate( d, e, f, a, b, c, dir ) &&
        !doesEdgeXySeparate( e, f, d, a, b, c, dir ) &&
        !doesEdgeXySeparate( f, d, e, a, b, c, dir );
}

/// checks whether triangle ABC and infinite line DE intersect
template <typename T>
bool doTriangleLineIntersect(
    const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c,
    const Vector3<T> & d, const Vector3<T> & e
)
{
    const auto dabe = mixed( d - e, a - e, b - e );
    const auto dbce = mixed( d - e, b - e, c - e );
    if ( dabe * dbce <= 0 )
        return false; // segment AC is located at one side of the plane DEB

    const auto dcae = mixed( d - e, c - e, a - e );
    if ( dbce * dcae <= 0 )
        return false; // segment AB is located at one side of the plane DEC

    if ( dcae * dabe <= 0 )
        return false; // segment BC is located at one side of the plane DEA

    return true;
}

/// checks whether triangle ABC and segment DE intersect
template <typename T>
bool doTriangleSegmentIntersect(
    const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c,
    const Vector3<T> & d, const Vector3<T> & e
)
{
    const auto abcd = mixed( a - d, b - d, c - d );
    const auto abce = mixed( a - e, b - e, c - e );
    if ( abcd * abce >= 0 )
        return false; // segment DE is located at one side of the plane ABC
    return doTriangleLineIntersect( a, b, c, d, e );
}

/// this function input should have intersection
template <typename T>
Vector3<T> findTriangleSegmentIntersection(
    const Vector3<T>& a, const Vector3<T>& b, const Vector3<T>& c,
    const Vector3<T>& d, const Vector3<T>& e
)
{
    const auto abcd = std::abs( mixed( a - d, b - d, c - d ) );
    const auto abce = std::abs( mixed( a - e, b - e, c - e ) );
    auto r = std::clamp( abcd / ( abcd + abce ), T( 0 ), T( 1 ) );
    return r * e + ( 1 - r ) * d;
}

/// returns any intersection point of triangle ABC and triangle DEF, if they intersects
/// returns nullopt if they do not intersect (also might return nullopt in degenerated cases)
template <typename T>
std::optional<Vector3<T>> findTriangleTriangleIntersection(
    const Vector3<T>& a, const Vector3<T>& b, const Vector3<T>& c,
    const Vector3<T>& d, const Vector3<T>& e, const Vector3<T>& f )
{
    if ( doTriangleSegmentIntersect( a, b, c, d, e ) )
        return findTriangleSegmentIntersection( a, b, c, d, e );
    if ( doTriangleSegmentIntersect( a, b, c, e, f ) )
        return findTriangleSegmentIntersection( a, b, c, e, f );
    if ( doTriangleSegmentIntersect( a, b, c, f, d ) )
        return findTriangleSegmentIntersection( a, b, c, f, d );

    if ( doTriangleSegmentIntersect( d, e, f, a, b ) )
        return findTriangleSegmentIntersection( d, e, f, a, b );
    if ( doTriangleSegmentIntersect( d, e, f, b, c ) )
        return findTriangleSegmentIntersection( d, e, f, b, c );
    if ( doTriangleSegmentIntersect( d, e, f, c, a ) )
        return findTriangleSegmentIntersection( d, e, f, c, a );
    return std::nullopt;
}

template <typename T>
std::optional<TriIntersectResult> rayTriangleIntersect( const Vector3<T>& oriA, const Vector3<T>& oriB, const Vector3<T>& oriC,
    const IntersectionPrecomputes<T>& prec )
{
    const T& Sx = prec.Sx;
    const T& Sy = prec.Sy;
    const T& Sz = prec.Sz;

    const T Ax = oriA[prec.idxX] - Sx * oriA[prec.maxDimIdxZ];
    const T Ay = oriA[prec.idxY] - Sy * oriA[prec.maxDimIdxZ];
    const T Bx = oriB[prec.idxX] - Sx * oriB[prec.maxDimIdxZ];
    const T By = oriB[prec.idxY] - Sy * oriB[prec.maxDimIdxZ];
    const T Cx = oriC[prec.idxX] - Sx * oriC[prec.maxDimIdxZ];
    const T Cy = oriC[prec.idxY] - Sy * oriC[prec.maxDimIdxZ];

    // due to fused multiply-add (FMA): (A*B-A*B) can be different from zero, so we need epsilon
    const T eps = std::numeric_limits<T>::epsilon() * std::max( { Ax, Bx, Cx, Ay, By, Cy } );
    T U = Cx * By - Cy * Bx;
    T V = Ax * Cy - Ay * Cx;
    T W = Bx * Ay - By * Ax;

    if( U < -eps || V < -eps || W < -eps )
    {
        if( U > eps || V > eps || W > eps )
        {
            // U,V,W have clearly different signs, so the ray misses the triangle
            return std::nullopt;
        }
    }

    T det = U + V + W;
    if( det == T( 0 ) )
        return std::nullopt;
    const T Az = Sz * oriA[prec.maxDimIdxZ];
    const T Bz = Sz * oriB[prec.maxDimIdxZ];
    const T Cz = Sz * oriC[prec.maxDimIdxZ];
    const T t = U * Az + V * Bz + W * Cz;

    auto invDet = T( 1 ) / det;
    return TriIntersectResult( float( V * invDet ), float( W * invDet ), float( t * invDet ) );
}

MR_BIND_TEMPLATE( std::optional<TriIntersectResult> rayTriangleIntersect( const Vector3<float >& oriA, const Vector3<float >& oriB, const Vector3<float >& oriC, const IntersectionPrecomputes<float >& prec ) )
MR_BIND_TEMPLATE( std::optional<TriIntersectResult> rayTriangleIntersect( const Vector3<double>& oriA, const Vector3<double>& oriB, const Vector3<double>& oriC, const IntersectionPrecomputes<double>& prec ) )

template <typename T>
std::optional<TriIntersectResult> rayTriangleIntersect( const Vector3<T>& oriA, const Vector3<T>& oriB, const Vector3<T>& oriC,
    const Vector3<T>& dir )
{
    const IntersectionPrecomputes<T> prec( dir );
    return rayTriangleIntersect( oriA, oriB, oriC, prec );
}

MR_BIND_TEMPLATE( std::optional<TriIntersectResult> rayTriangleIntersect( const Vector3<float >& oriA, const Vector3<float >& oriB, const Vector3<float >& oriC, const Vector3<float >& dir ) )
MR_BIND_TEMPLATE( std::optional<TriIntersectResult> rayTriangleIntersect( const Vector3<double>& oriA, const Vector3<double>& oriB, const Vector3<double>& oriC, const Vector3<double>& dir ) )

/// returns true if ABC and DEF overlaps or touches
template<typename T>
bool doTrianglesOverlap( const Vector2<T>& a, const Vector2<T>& b, const Vector2<T>& c, const Vector2<T>& d, const Vector2<T>& e, const Vector2<T>& f )
{
    // TODO: probably some of the checks are excessive?

    // check if AB intersects any of DEF sides
    if ( doSegmentsIntersect( LineSegm<Vector2<T>>{ a,b }, LineSegm<Vector2<T>>{ d,e } ) )
        return true;
    if ( doSegmentsIntersect( LineSegm<Vector2<T>>{ a,b }, LineSegm<Vector2<T>>{ d,f } ) )
        return true;
    if ( doSegmentsIntersect( LineSegm<Vector2<T>>{ a,b }, LineSegm<Vector2<T>>{ e,f } ) )
        return true;

    // check if AC intersects any of DEF sides
    if ( doSegmentsIntersect( LineSegm<Vector2<T>>{ a,c }, LineSegm<Vector2<T>>{ d,e } ) )
        return true;
    if ( doSegmentsIntersect( LineSegm<Vector2<T>>{ a,c }, LineSegm<Vector2<T>>{ d,f } ) )
        return true;
    if ( doSegmentsIntersect( LineSegm<Vector2<T>>{ a,c }, LineSegm<Vector2<T>>{ e,f } ) )
        return true;

    // check if BC intersects any of DEF sides
    if ( doSegmentsIntersect( LineSegm<Vector2<T>>{ b,c }, LineSegm<Vector2<T>>{ d,e } ) )
        return true;
    if ( doSegmentsIntersect( LineSegm<Vector2<T>>{ b,c }, LineSegm<Vector2<T>>{ d,f } ) )
        return true;
    if ( doSegmentsIntersect( LineSegm<Vector2<T>>{ b,c }, LineSegm<Vector2<T>>{ e,f } ) )
        return true;

    // no sides intersection:
    // either ABC fully inside DEF or vice versa
    if ( isPointInTriangle( a, d, e, f ) )
        return true;
    if ( isPointInTriangle( d, a, b, c ) )
        return true;

    return false;
}

MR_BIND_TEMPLATE( bool doTrianglesOverlap( const Vector2<float>& a, const Vector2<float>& b, const Vector2<float>& c, const Vector2<float>& d, const Vector2<float>& e, const Vector2<float>& f ) )
MR_BIND_TEMPLATE( bool doTrianglesOverlap( const Vector2<double>& a, const Vector2<double>& b, const Vector2<double>& c, const Vector2<double>& d, const Vector2<double>& e, const Vector2<double>& f ) )

/// \}

} // namespace MR
