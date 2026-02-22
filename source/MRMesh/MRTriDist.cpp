#include "MRTriDist.h"
#include "MRTwoLineSegmDist.h"

namespace MR
{

namespace
{

// based on the code by E. Larsen from University of N. Carolina

template<class T>
TriTriDistanceResult<T> findDistanceT( const Triangle3<T>& a, const Triangle3<T>& b )
{
    TriTriDistanceResult<T> res;

    // Compute vectors along the 6 sides
    Vector3<T> VEC;

    const Vector3<T> Sv[3] =
    {
        a[1] - a[0],
        a[2] - a[1],
        a[0] - a[2]
    };

    const Vector3<T> Tv[3] =
    {
        b[1] - b[0],
        b[2] - b[1],
        b[0] - b[2]
    };

    // For each edge pair, the vector connecting the closest points
    // of the edges defines a slab (parallel planes at head and tail
    // enclose the slab). If we can show that the off-edge vertex of
    // each triangle is outside of the slab, then the closest points
    // of the edges are the closest points for the triangles.
    // Even if these tests fail, it may be helpful to know the closest
    // points found, and whether the triangles were shown disjoint

    Vector3<T> V, Z, minP, minQ;
    T mindd;
    int shown_disjoint = 0;

    mindd = ( a[0] - b[0] ).lengthSq() + 1;  // Set first minimum safely high

    for ( int i = 0; i < 3; i++ )
    {
        for ( int j = 0; j < 3; j++ )
        {
            // Find closest points on edges i & j, plus the
            // vector (and distance squared) between these points

            static constexpr int next[3] = { 1, 2, 0 };
            const auto sd = findTwoLineSegmClosestPoints( { a[i], a[next[i]] }, { b[j], b[next[j]] } );
            res.a = sd.a;
            res.b = sd.b;
            VEC = sd.dir;

            V = res.b - res.a;
            T dd = dot( V, V );

            // Verify this closest point pair only if the distance
            // squared is less than the minimum found thus far.

            if ( dd <= mindd )
            {
                minP = res.a;
                minQ = res.b;
                mindd = dd;

                Z = a[( i + 2 ) % 3] - res.a;
                T s = dot( Z, VEC );
                Z = b[( j + 2 ) % 3] - res.b;
                T t = dot( Z, VEC );

                if ( ( s <= 0 ) && ( t >= 0 ) )
                {
                    res.distSq = dd;
                    return res;
                }

                T p = dot( V, VEC );

                if ( s < 0 ) s = 0;
                if ( t > 0 ) t = 0;
                if ( ( p - s + t ) > 0 ) shown_disjoint = 1;
            }
        }
    }

    // No edge pairs contained the closest points.
    // either:
    // 1. one of the closest points is a vertex, and the
    //    other point is interior to a face.
    // 2. the triangles are overlapping.
    // 3. an edge of one triangle is parallel to the other's face. If
    //    cases 1 and 2 are not true, then the closest points from the 9
    //    edge pairs checks above can be taken as closest points for the
    //    triangles.
    // 4. possibly, the triangles were degenerate.  When the
    //    triangle points are nearly colinear or coincident, one
    //    of above tests might fail even though the edges tested
    //    contain the closest points.

    // First check for case 1

    Vector3<T> Sn = cross( Sv[0], Sv[1] ); // Compute normal to a triangle
    T Snl = dot( Sn, Sn );      // Compute square of length of normal

    // If cross product is long enough,

    if ( Snl > 1e-15 )
    {
        // Get projection lengths of b points

        T Tp[3];

        V = a[0] - b[0];
        Tp[0] = dot( V, Sn );

        V = a[0] - b[1];
        Tp[1] = dot( V, Sn );

        V = a[0] - b[2];
        Tp[2] = dot( V, Sn );

        // If Sn is a separating direction,
        // find point with smallest projection

        int point = -1;
        if ( ( Tp[0] > 0 ) && ( Tp[1] > 0 ) && ( Tp[2] > 0 ) )
        {
            if ( Tp[0] < Tp[1] ) point = 0; else point = 1;
            if ( Tp[2] < Tp[point] ) point = 2;
        }
        else if ( ( Tp[0] < 0 ) && ( Tp[1] < 0 ) && ( Tp[2] < 0 ) )
        {
            if ( Tp[0] > Tp[1] ) point = 0; else point = 1;
            if ( Tp[2] > Tp[point] ) point = 2;
        }

        // If Sn is a separating direction,

        if ( point >= 0 )
        {
            shown_disjoint = 1;

            // Test whether the point found, when projected onto the
            // other triangle, lies within the face.

            V = b[point] - a[0];
            Z = cross( Sn, Sv[0] );
            if ( dot( V, Z ) > 0 )
            {
                V = b[point] - a[1];
                Z = cross( Sn, Sv[1] );
                if ( dot( V, Z ) > 0 )
                {
                    V = b[point] - a[2];
                    Z = cross( Sn, Sv[2] );
                    if ( dot( V, Z ) > 0 )
                    {
                        // b[point] passed the test - it's a closest point for
                        // the b triangle; the other point is on the face of a

                        res.a = b[point] + Sn * Tp[point] / Snl;
                        res.b = b[point];
                        res.distSq = distanceSq( res.a, res.b );
                        return res;
                    }
                }
            }
        }
    }

    Vector3<T> Tn = cross( Tv[0], Tv[1] );
    T Tnl = dot( Tn, Tn );

    if ( Tnl > 1e-15 )
    {
        T Sp[3];

        V = b[0] - a[0];
        Sp[0] = dot( V, Tn );

        V = b[0] - a[1];
        Sp[1] = dot( V, Tn );

        V = b[0] - a[2];
        Sp[2] = dot( V, Tn );

        int point = -1;
        if ( ( Sp[0] > 0 ) && ( Sp[1] > 0 ) && ( Sp[2] > 0 ) )
        {
            if ( Sp[0] < Sp[1] ) point = 0; else point = 1;
            if ( Sp[2] < Sp[point] ) point = 2;
        }
        else if ( ( Sp[0] < 0 ) && ( Sp[1] < 0 ) && ( Sp[2] < 0 ) )
        {
            if ( Sp[0] > Sp[1] ) point = 0; else point = 1;
            if ( Sp[2] > Sp[point] ) point = 2;
        }

        if ( point >= 0 )
        {
            shown_disjoint = 1;

            V = a[point] - b[0];
            Z = cross( Tn, Tv[0] );
            if ( dot( V, Z ) > 0 )
            {
                V = a[point] - b[1];
                Z = cross( Tn, Tv[1] );
                if ( dot( V, Z ) > 0 )
                {
                    V = a[point] - b[2];
                    Z = cross( Tn, Tv[2] );
                    if ( dot( V, Z ) > 0 )
                    {
                        res.a = a[point];
                        res.b = a[point] + Tn * Sp[point] / Tnl;
                        res.distSq = distanceSq( res.a, res.b );
                        return res;
                    }
                }
            }
        }
    }

    // Case 1 can't be shown.
    // If one of these tests showed the triangles disjoint,
    // we assume case 3 or 4, otherwise we conclude case 2,
    // that the triangles overlap.

    if ( shown_disjoint )
    {
        res.a = minP;
        res.b = minQ;
        res.distSq = mindd;
        return res;
    }

    res.distSq = 0;
    return res;
}

} // anonymous namespace

TriTriDistanceResultf findDistance( const Triangle3f& a, const Triangle3f& b )
{
    return findDistanceT( a, b );
}

TriTriDistanceResultd findDistance( const Triangle3d& a, const Triangle3d& b )
{
    return findDistanceT( a, b );
}

float triDist( Vector3f & p, Vector3f & q, const Vector3f s[3], const Vector3f t[3] )
{
    const auto td = findDistance( { s[0], s[1], s[2] }, { t[0], t[1], t[2] } );
    p = td.a;
    q = td.b;
    return td.distSq;
}

float triDist( Vector3f & p, Vector3f & q, const std::array<Vector3f, 3> & s, const std::array<Vector3f, 3> & t )
{
    const auto td = findDistance( s, t );
    p = td.a;
    q = td.b;
    return td.distSq;
}

void segPoints( Vector3f & VEC,
          Vector3f & X, Vector3f & Y,
          const Vector3f & P, const Vector3f & A,
          const Vector3f & Q, const Vector3f & B)
{
    const auto sd = findTwoLineSegmClosestPoints( { P, P + A }, { Q, Q + B } );
    X = sd.a;
    Y = sd.b;
    VEC = sd.dir;
}

} // namespace MR
