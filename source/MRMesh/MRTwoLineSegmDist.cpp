#include "MRTwoLineSegmDist.h"

namespace MR
{

namespace
{

// Implemented from an algorithm described in
//
// Vladimir J. Lumelsky,
// On fast computation of distance between line segments.
// In Information Processing Letters, no. 21, pages 55-61, 1985.
template<class T>
TwoLineSegmClosestPoints<T> findTwoLineSegmClosestPointsT( const LineSegm3<T>& a, const LineSegm3<T>& b )
{
    TwoLineSegmClosestPoints<T> res;

    const auto adir = a.dir();
    const auto bdir = b.dir();

    const auto aa = dot( adir, adir );
    const auto bb = dot( bdir, bdir );
    const auto ab = dot( adir, bdir );
    const auto denom = aa * bb - ab * ab;

           auto d = b.a - a.a;
    const auto ad = dot( adir, d );
    const auto bd = dot( bdir, d );

    // compute t for the closest point on ray a to ray b
    // t parameterizes ray a
    auto t = ( ad * bb - bd * ab ) / denom;

    // clamp result so t is on the segment a.a,adir

    if ( ( t < 0 ) || std::isnan( t ) ) t = 0; else if ( t > 1 ) t = 1;

    // find u for point on ray b closest to point ad t
    // u parameterizes ray b
    auto u = ( t * ab - bd ) / bb;

    // if u is on segment b.a,bdir, t and u correspond to
    // closest points, otherwise, clamp u, recompute and
    // clamp t

    if ( ( u <= 0 ) || std::isnan( u ) )
    {
        res.b = b.a;

        t = ad / aa;

        if ( ( t <= 0 ) || std::isnan( t ) )
        {
            res.a = a.a;
            res.dir = b.a - a.a;
        }
        else if ( t >= 1 )
        {
            res.a = a.a + adir;
            res.dir = b.a - res.a;
        }
        else
        {
            res.a = a.a + adir * t;
            auto tmp = cross( d, adir );
            res.dir = cross( adir, tmp );
        }
    }
    else if ( u >= 1 )
    {
        res.b = b.a + bdir;

        t = ( ab + ad ) / aa;

        if ( ( t <= 0 ) || std::isnan( t ) )
        {
            res.a = a.a;
            res.dir = res.b - a.a;
        }
        else if ( t >= 1 )
        {
            res.a = a.a + adir;
            res.dir = res.b - res.a;
        }
        else
        {
            res.a = a.a + adir * t;
            d = res.b - a.a;
            auto tmp = cross( d, adir );
            res.dir = cross( adir, tmp );
        }
    }
    else
    {
        res.b = b.a + bdir * u;

        if ( ( t <= 0 ) || std::isnan( t ) )
        {
            res.a = a.a;
            auto tmp = cross( d, bdir );
            res.dir = cross( bdir, tmp );
        }
        else if ( t >= 1 )
        {
            res.a = a.a + adir;
            d = b.a - res.a;
            auto tmp = cross( d, bdir );
            res.dir = cross( bdir, tmp );
        }
        else
        {
            res.a = a.a + adir * t;
            res.dir = cross( adir, bdir );
            if ( dot( res.dir, d ) < 0 )
                res.dir = -res.dir;
        }
    }

    return res;
}

} // anonymous namespace

TwoLineSegmClosestPointsf findTwoLineSegmClosestPoints( const LineSegm3f& a, const LineSegm3f& b )
{
    return findTwoLineSegmClosestPointsT( a, b );
}

TwoLineSegmClosestPointsd findTwoLineSegmClosestPoints( const LineSegm3d& a, const LineSegm3d& b )
{
    return findTwoLineSegmClosestPointsT( a, b );
}

} //namespace MR
