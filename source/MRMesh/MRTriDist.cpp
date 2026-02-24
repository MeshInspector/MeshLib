#include "MRTriDist.h"
#include "MRTwoLineSegmDist.h"
#include <optional>

namespace MR
{

namespace
{

// based on the code by E. Larsen from University of N. Carolina

// a. tests whether normal of a-triangle is a separating direction, then sets overlap flag to false;
// b. if separating direction confirmed, then additionally tests whether the closest of b-vertices to a-triangle
//       projects in the interior of a-triangle, then returns them as the closest pair of points.
template<class T>
std::optional<TriTriDistanceResult<T>> projectBonA( const Triangle3<T>& a, const Triangle3<T>& b, bool& overlap )
{
    // vectors of a-edges:
    const Vector3<T> av[3] =
    {
        a[1] - a[0],
        a[2] - a[1],
        a[0] - a[2]
    };

    const Vector3<T> an = cross( av[0], av[1] ); // Compute normal to a triangle
    const T anSq = an.lengthSq();

    // If a-triangles is not degenerate
    if ( anSq > 0 )
    {
        // projections of b-points on -an direction
        const T bp[3] =
        {
            dot( a[0] - b[0], an ),
            dot( a[0] - b[1], an ),
            dot( a[0] - b[2], an )
        };

        // if all bps have the same sign, then an is a separating direction;
        // find the point with the smallest projection
        int point = -1;
        if ( ( bp[0] >= 0 ) && ( bp[1] >= 0 ) && ( bp[2] >= 0 ) )
        {
            if ( bp[0] < bp[1] ) point = 0; else point = 1;
            if ( bp[2] < bp[point] ) point = 2;
        }
        else if ( ( bp[0] <= 0 ) && ( bp[1] <= 0 ) && ( bp[2] <= 0 ) )
        {
            if ( bp[0] > bp[1] ) point = 0; else point = 1;
            if ( bp[2] > bp[point] ) point = 2;
        }

        // If an is a separating direction,
        if ( point >= 0 )
        {
            overlap = false;

            // Test whether the point found, when projected onto the
            // other triangle, lies within the face.

            if ( mixed( b[point] - a[0], an, av[0] ) > 0
              && mixed( b[point] - a[1], an, av[1] ) > 0
              && mixed( b[point] - a[2], an, av[2] ) > 0 )
            {
                // b[point] passed the test - it's the closest point for
                // the b triangle; the other point is on the face of a
                TriTriDistanceResult<T> res;
                res.a = b[point] + an * bp[point] / anSq;
                res.b = b[point];
                res.distSq = distanceSq( res.a, res.b );
                res.overlap = false;
                return res;
            }
        }
    }
    return std::nullopt;
}

template<class T>
TriTriDistanceResult<T> findTriTriDistanceT( const Triangle3<T>& a, const Triangle3<T>& b )
{
    // For each edge pair, the vector connecting the closest points
    // of the edges defines a slab (parallel planes at head and tail
    // enclose the slab). If we can show that the off-edge vertex of
    // each triangle is outside of the slab, then the closest points
    // of the edges are the closest points for the triangles.
    // Even if these tests fail, it may be helpful to know the closest
    // points found, and whether the triangles were shown disjoint

    // the distance between the triangles is not more than the distance between two of their points
    TriTriDistanceResult<T> res
    {
        .a = a[0],
        .b = b[0],
        .distSq = distanceSq( a[0], b[0] )
    };

    for ( int i = 0; i < 3; i++ )
    {
        for ( int j = 0; j < 3; j++ )
        {
            // Find closest points on edges { a[i], a[next[i]] } & { b[j], b[next[j]] }, plus the
            // vector (and distance squared) between these points

            static constexpr int prev[3] = { 2, 0, 1 };
            static constexpr int next[3] = { 1, 2, 0 };
            const auto sd = findTwoLineSegmClosestPoints( { a[i], a[next[i]] }, { b[j], b[next[j]] } );
            const T dd = distanceSq( sd.a, sd.b );

            // Verify this closest point pair only if the distance
            // squared is less than the minimum found thus far.

            if ( dd <= res.distSq ) // no strictly less, to set res.overlap
            {
                res.a = sd.a;
                res.b = sd.b;
                res.distSq = dd;

                // a[prev[i]] and b[prev[j]] are remaining vertices of the triangles
                // on top of the vertices from the considered edges
                T s = dot( a[prev[i]] - res.a, sd.dir );
                T t = dot( b[prev[j]] - res.b, sd.dir );

                // if the remaining points are further along sd.dir than the considered edges
                if ( ( s <= 0 ) && ( t >= 0 ) )
                {
                    res.overlap = false;
                    return res;
                }

                // the distance along sd.dir between the considered edges
                const T p = dot( res.b - res.a, sd.dir );

                if ( s < 0 ) s = 0;
                if ( t > 0 ) t = 0;

                // sd.dir is a separating direction
                if ( ( p - s + t ) >= 0 ) res.overlap = false;
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
    if ( auto maybeRes = projectBonA( a, b, res.overlap ) )
        return *maybeRes;

    if ( auto maybeRes = projectBonA( b, a, res.overlap ) )
    {
        std::swap( maybeRes->a, maybeRes->b );
        return *maybeRes;
    }

    // Case 1 can't be shown.
    // If one of these tests showed the triangles disjoint,
    // we assume case 3 or 4, otherwise we conclude case 2,
    // that the triangles overlap.

    if ( res.overlap )
        res.distSq = 0; // triangles are colliding (overlapping)

    return res;
}

} // anonymous namespace

TriTriDistanceResultf findTriTriDistance( const Triangle3f& a, const Triangle3f& b )
{
    return findTriTriDistanceT( a, b );
}

TriTriDistanceResultd findTriTriDistance( const Triangle3d& a, const Triangle3d& b )
{
    return findTriTriDistanceT( a, b );
}

float triDist( Vector3f & p, Vector3f & q, const Vector3f s[3], const Vector3f t[3] )
{
    const auto td = findTriTriDistance( { s[0], s[1], s[2] }, { t[0], t[1], t[2] } );
    p = td.a;
    q = td.b;
    return td.distSq;
}

float triDist( Vector3f & p, Vector3f & q, const std::array<Vector3f, 3> & s, const std::array<Vector3f, 3> & t )
{
    const auto td = findTriTriDistance( s, t );
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
