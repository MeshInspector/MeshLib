#pragma once

#include "MRVector3.h"
#include "MRIntersectionPrecomputes.h"
#include "MRTriPoint.h"

namespace MR
{

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

// checks whether triangles ABC and DEF intersect
template <typename T>
bool doTrianglesIntersect(
    Vector3<T> a, Vector3<T> b, Vector3<T> c,
    Vector3<T> d, Vector3<T> e, Vector3<T> f
)
{
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

    const auto acdf = mixed( a - f, c - f, d - f );

    if ( acde * acdf < 0 )
        return true; // AC segment penetrates triangle DEF since points E and F are at distinct sides of ACD

    if ( abdf * acdf < 0 )
        return true; // DF segment penetrates triangle ABC since points B and C are at distinct sides of ADF

    return false;
}

// returns true if a plane containing edge XY separates point Z from triangle UVW
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

// checks whether triangles ABC and DEF intersect;
// performs more checks to avoid false positives of simple doTrianglesIntersect
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

// checks whether triangle ABC and segment DE intersect
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

// this function input should have intersection
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

template <typename T>
std::optional<TriIntersectResult> rayTriangleIntersect_( const Vector3<T>& oriA, const Vector3<T>& oriB, const Vector3<T>& oriC,
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

    T U = Cx * By - Cy * Bx;
    T V = Ax * Cy - Ay * Cx;
    T W = Bx * Ay - By * Ax;

    if( U < T( 0 ) || V < T( 0 ) || W < T( 0 ) )
    {
        if( U > T( 0 ) || V > T( 0 ) || W > T( 0 ) )
        {
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

inline std::optional<TriIntersectResult> rayTriangleIntersect( const Vector3f& oriA, const Vector3f& oriB, const Vector3f& oriC,
    const IntersectionPrecomputes<float>& prec )
{
    return rayTriangleIntersect_( oriA, oriB, oriC, prec );
}
inline std::optional<TriIntersectResult> rayTriangleIntersect( const Vector3f& oriA, const Vector3f& oriB, const Vector3f& oriC,
    const Vector3f& dir )
{
    const IntersectionPrecomputes<float> prec( dir );
    return rayTriangleIntersect_( oriA, oriB, oriC, prec );
}

inline std::optional<TriIntersectResult> rayTriangleIntersect( const Vector3d& oriA, const Vector3d& oriB, const Vector3d& oriC,
    const IntersectionPrecomputes<double>& prec )
{
    return rayTriangleIntersect_( oriA, oriB, oriC, prec );
}

inline std::optional<TriIntersectResult> rayTriangleIntersect( const Vector3d& oriA, const Vector3d& oriB, const Vector3d& oriC,
    const Vector3d& dir )
{
    const IntersectionPrecomputes<double> prec( dir );
    return rayTriangleIntersect_( oriA, oriB, oriC, prec );
}

} // namespace MR
