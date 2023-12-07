#pragma once

#include "MRPlane3.h"
#include "MRLine3.h"
#include "MRLineSegm.h"
#include "MRVector2.h"
#include "MRBox.h"
#include "MRSphere.h"
#include <optional>

namespace MR
{

/// \defgroup IntersectionGroup Intersection
/// \ingroup MathGroup
/// \{

/// finds an intersection between a plane1 and a plane2
/// \param plane1,plane2 should be normalized for check parallelism
/// \return nullopt if they are parallel (even if they match)
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

/// finds an intersection between a plane and a line
/// \param plane,line should be normalized for check parallelism
/// \return nullopt if they are parallel (even line belongs to plane)
template<typename T>
std::optional<Vector3<T>> intersection( const Plane3<T>& plane, const Line3<T>& line,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    const auto den = dot( plane.n, line.d );
    if ( std::abs(den) < errorLimit )
        return {};
    return line.p + ( plane.d - dot( plane.n, line.p ) ) / den * line.d;
}

/// finds an intersection between a line1 and a line2
/// \param line1,line2 should be normalized for check parallelism
/// \return nullopt if they are not intersect (even if they match)
template<typename T>
std::optional<Vector3<T>> intersection( const Line3<T>& line1, const Line3<T>& line2,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    const auto crossDir = cross( line1.d, line2.d );
    if ( crossDir.lengthSq() < errorLimit * errorLimit )
        return {};

    const auto p1 = dot( crossDir, line1.p );
    const auto p2 = dot( crossDir, line2.p );
    if ( std::abs( p1 - p2 ) >= errorLimit )
        return {};

    const auto n2 = cross( line2.d, crossDir );
    const T den = dot( line1.d, n2 );
    if ( den == 0 ) // check for calculation
        return {};
    return line1.p + dot( ( line2.p - line1.p ), n2 ) / den * line1.d;
}

/// finds an intersection between a segm1 and a segm2
/// \return nullopt if they don't intersect (even if they match)
inline std::optional<Vector2f> intersection( const LineSegm2f& segm1, const LineSegm2f& segm2 )
{
    auto avec = segm1.b - segm1.a;
    if ( cross( avec, segm2.a - segm1.a ) * cross( segm2.b - segm1.a, avec ) <= 0 )
        return {};
    auto bvec = segm2.b - segm2.a;
    auto cda = cross( bvec, segm1.a - segm2.a );
    auto cbd = cross( segm1.b - segm2.a, bvec );
    if ( cda * cbd <= 0 )
        return {};
    return ( segm1.b * cda + segm1.a * cbd ) / ( cda + cbd );
}

/// finds squared distance between a plane1 and a plane2
/// \return nullopt if they intersect
template<typename T>
std::optional<T> distanceSq( const Plane3<T>& plane1, const Plane3<T>& plane2,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    const auto crossDir = cross( plane1.n, plane2.n );

    if ( crossDir.lengthSq() >= errorLimit * errorLimit )
        return {};

    return ( plane2.n * plane2.d - plane1.n * plane1.d ).lengthSq();
}

/// finds distance between a plane1 and a plane2
/// \return nullopt if they intersect
template<typename T>
std::optional<T> distance( const Plane3<T>& plane1, const Plane3<T>& plane2,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    std::optional<T> res = distanceSq( plane1, plane2, errorLimit );
    if ( res )
        *res = std::sqrt( *res );
    return res;
}

/// finds distance between a plane and a line;
/// \return nullopt if they intersect
template<typename T>
std::optional<T> distance( const Plane3<T>& plane, const Line3<T>& line,
    T errorLimit = std::numeric_limits<T>::epsilon() * T( 20 ) )
{
    const auto den = dot( plane.n, line.d );
    if ( std::abs( den ) >= errorLimit )
        return {};

    return std::abs( dot( line.p, plane.n ) - plane.d );
}

/// finds the closest points between two lines in 3D;
/// for parallel lines the selection is arbitrary;
/// \return two equal points if the lines intersect
template<typename T>
LineSegm3<T> closestPoints( const Line3<T>& line1, const Line3<T>& line2 )
{
    const auto d11 = line1.d.lengthSq();
    const auto d12 = dot( line1.d, line2.d );
    const auto d22 = line2.d.lengthSq();
    const auto det = d12 * d12 - d11 * d22;
    if ( det == 0 )
    {
        // lines are parallel
        return { line1.p, line2.project( line1.p ) };
    }

    const auto dp = line2.p - line1.p;
    const auto x = dot( dp, line1.d ) / det;
    const auto y = dot( dp, line2.d ) / det;
    const auto a = d12 * y - d22 * x;
    const auto b = d11 * y - d12 * x;
    return { line1( a ), line2( b ) };
}

/// finds the closest points between an infinite line and finite line segment in 3D;
/// for parallel lines the selection is arbitrary;
/// \return two equal points if the lines intersect
template<typename T>
LineSegm3<T> closestPoints( const Line3<T>& ln, const LineSegm3<T>& ls )
{
    const auto d11 = ln.d.lengthSq();
    const auto d12 = dot( ln.d, ls.dir() );
    const auto d22 = ls.lengthSq();
    const auto det = d12 * d12 - d11 * d22;
    if ( det == 0 ) // lines are parallel
        return { ln.project( ls.a ), ls.a };

    const auto dp = ls.a - ln.p;
    const auto x = dot( dp, ln.d ) / det;
    const auto y = dot( dp, ls.dir() ) / det;
    const auto b = d11 * y - d12 * x;
    if ( b <= 0 )
        return { ln.project( ls.a ), ls.a };
    if ( b >= 1 )
        return { ln.project( ls.b ), ls.b };
    const auto a = d12 * y - d22 * x;
    return { ln( a ), ls( b ) };
}

/// finds the closest points between a line and a box wireframe (not solid) in 3D
template<typename T>
LineSegm3<T> closestPoints( const Line3<T>& line, const Box3<T> & box )
{
    LineSegm3<T> res;
    const auto dd = line.d.lengthSq();
    if ( dd <= 0 )
    {
        res.a = line.p;
        res.b = box.getBoxClosestPointTo( res.a );
        return res;
    }
    const auto rdd = 1 / dd;

    T bestDistSq = std::numeric_limits<T>::max();

    static constexpr int otherDir[3][2] = { { 1, 2 }, { 2, 0 }, { 0, 1 } };
    for ( int iDir = 0; iDir < 3; ++iDir )
    {
        // consider box edges parallel to unit vector #iDir
        Vector3<T> q[4] = { box.min, box.min, box.min, box.min };
        {
            const int iDir1 = otherDir[iDir][0];
            const int iDir2 = otherDir[iDir][1];

            q[1][iDir2] = box.max[iDir2];

            q[2][iDir1] = box.max[iDir1];
            q[2][iDir2] = box.max[iDir2];

            q[3][iDir1] = box.max[iDir1];
        }

        const auto e = box.max[iDir] - box.min[iDir];
        const auto ee = e * e;
        const auto db = line.d[iDir] * e;
        const auto denom = dd * ee - db * db;
        const bool par = denom <= 0; // line is parallel to box edge
        const auto rdenom = par ? 0 : 1 / denom;
        for ( int j = 0; j < 4; ++j )
        {
            LineSegm3<T> cand;
            if ( par )
            {
                cand.a = line.p;
                cand.a[iDir] = q[j][iDir];
                cand.b = q[j];
            }
            else
            {
                const auto s = q[j] - line.p;
                const auto dt = dot( line.d, s );
                const auto bt = s[iDir] * e;

                // t is line parameter: 0 - line.p, 1 - line.p + line.d
                const auto t = ( dt * ee - bt * db ) * rdenom;
                assert( !std::isnan( t ) );

                // u is box edge parameter, find the point closest to line(t)
                const auto u = ( t * db - bt ) / ee;
                assert( !std::isnan( u ) );

                if ( u <= 0 )
                {
                    cand.a = line( dt * rdd );
                    cand.b = q[j];
                }
                else if ( u >= 1 )
                {
                    cand.a = line( ( db + dt ) * rdd );
                    cand.b = q[j];
                    cand.b[iDir] = box.max[iDir];
                }
                else 
                {
                    cand.a = line( t );
                    cand.b = q[j];
                    cand.b[iDir] += e * u;
                }
            }
            const auto distSq = cand.lengthSq();
            if ( distSq < bestDistSq )
            {
                bestDistSq = distSq;
                res = cand;
            }
        }
    }
    return res;
}

/// finds intersection points between a line and a sphere;
/// if found then returns parameters on the line
template<typename V>
auto intersection( const Line<V>& line, const Sphere<V>& sphere )
{
    using T = typename V::ValueType;
    std::optional<std::pair<T,T>> res;
    const auto p = line.p - sphere.center;
    const auto d = line.d;
    const auto dd = dot( d, d );
    const auto pd = dot( p, d );
    const auto des4 = sqr( pd ) - dd * ( dot( p, p ) - sqr( sphere.radius ) );
    if ( des4 < 0 )
        return res;
    const auto sqrtDes4 = std::sqrt( des4 );
    res.emplace();
    res->first = ( -sqrtDes4 - pd ) / dd;
    res->second = ( sqrtDes4 - pd ) / dd;
    return res;
}

/// \}

} // namespace MR
