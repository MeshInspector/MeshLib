#include "MRPolylineDistance.h"
#include "MRPolyline.h"
#include "MRPolylineProject.h"
#include "MRAffineXf3.h"
#include "MRBitSet.h"
#include "MRBitSetParallelFor.h"
#include "MRLineSegm.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include <optional>

namespace MR
{

PolylinePolylineDistanceResult findMaxDistanceOneWay( const Polyline3& a, const Polyline3& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    MR_TIMER;

    PolylinePolylineDistanceResult res;
    const auto& validVerts = b.topology.getValidVerts();
    const auto firstVert = validVerts.find_first();
    if ( !firstVert )
        return res;

    if ( !a.topology.lastNotLoneEdge().valid() )
    {
        // nothing to measure the distance to
        res.b = b.toEdgePoint( firstVert );
        res.distSq = maxDistanceSq;
        return res;
    }

    auto inA = [&]( VertId v ) { return rigidB2A ? ( *rigidB2A )( b.points[v] ) : b.points[v]; };

    struct VertDist
    {
        VertId v;
        float distSq = -1;
    };
    a.getAABBTree(); // prepare the tree before the parallel region
    // of several equally distant vertices, the one with the smallest id is taken
    const auto best = tbb::parallel_reduce
    (
        tbb::blocked_range( firstVert, validVerts.find_last() + 1 ),
        VertDist{},
        [&] ( const tbb::blocked_range<VertId>& range, VertDist init )
        {
            for ( VertId i = range.begin(); i < range.end(); ++i )
            {
                if ( !validVerts.test( i ) )
                    continue;
                const auto distSq = findProjectionOnPolyline( inA( i ), a, maxDistanceSq ).distSq;
                if ( distSq > init.distSq )
                    init = { i, distSq };
            }
            return init;
        },
        [] ( const VertDist& x, const VertDist& y ) { return y.distSq > x.distSq ? y : x; }
    );

    const auto proj = findProjectionOnPolyline( inA( best.v ), a, maxDistanceSq );
    if ( proj.valid() )
        res.a = a.toEdgePoint( EdgeId( proj.line ), proj.point );
    res.b = b.toEdgePoint( best.v );
    res.distSq = proj.distSq;
    return res;
}

PolylinePolylineDistanceResult findMaxDistance( const Polyline3& a, const Polyline3& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    std::optional<AffineXf3f> rigidA2B;
    if ( rigidB2A )
        rigidA2B = rigidB2A->inverse();
    auto res = findMaxDistanceOneWay( a, b, rigidB2A, maxDistanceSq );
    const auto ba = findMaxDistanceOneWay( b, a, rigidA2B ? &*rigidA2B : nullptr, maxDistanceSq );
    if ( ba.distSq > res.distSq )
        res = { ba.b, ba.a, ba.distSq };
    return res;
}

namespace
{

// a point of B in A's space, with the closest segment of A and the squared distance to it
struct LineCorner
{
    Vector3f p;
    float distSq = 0;
    UndirectedEdgeId e; // the closest segment of A
    float t = 0; // the position on the segment of B: 0 - in its origin, 1 - in its destination
};

bool hasEdges( const Polyline3& a )
{
    return a.topology.lastNotLoneEdge().valid();
}

LineCorner projectCorner( const Vector3f& p, float t, const Polyline3& a, float maxDistanceSq )
{
    if ( !hasEdges( a ) )
        return { p, maxDistanceSq, {}, t }; // nothing to measure the distance to
    const auto r = findProjectionOnPolyline( p, a, maxDistanceSq );
    return { p, r.distSq, r.line, t };
}

// all vertices of B projected on A
struct VertsOnPolyline
{
    Vector<LineCorner, VertId> corners;
    VertId maxVert; // the most distant vertex, or the first one farther than the limit; invalid if B has no vertices
    float maxSq = 0;
};

VertsOnPolyline projectVerts( const Polyline3& a, const Polyline3& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    VertsOnPolyline res;
    const auto& bVerts = b.topology.getValidVerts();
    if ( !bVerts.any() )
        return res;

    if ( hasEdges( a ) )
        a.getAABBTree(); // prepare the tree before the parallel region
    res.corners.resize( bVerts.find_last() + 1 );
    BitSetParallelFor( bVerts, [&]( VertId v )
    {
        res.corners[v] = projectCorner( rigidB2A ? ( *rigidB2A )( b.points[v] ) : b.points[v], 0, a, maxDistanceSq );
    } );
    // a vertex farther than the limit has distSq equal to the limit
    res.maxSq = -1;
    for ( auto v : bVerts )
    {
        if ( res.corners[v].distSq > res.maxSq )
        {
            res.maxVert = v;
            res.maxSq = res.corners[v].distSq;
        }
    }
    return res;
}

// bv - the vertices of B projected on A;
// startDistSq - the points of B not farther from A than this (plus tolerance) are of no interest, except for the most distant vertex
PolylinePolylineDistanceResult findMaxDistanceInsideSegments( const Polyline3& a, const Polyline3& b, float tolerance, float maxDistanceSq,
    const VertsOnPolyline& bv, float startDistSq )
{
    PolylinePolylineDistanceResult res;
    if ( !bv.maxVert )
        return res;

    using Segm = std::array<LineCorner, 2>;
    // no point of the segment is farther from A than from one segment of A, and the latter distance is maximal in an end
    auto upperBoundSq = [&]( const Segm& s )
    {
        float ubSq = FLT_MAX;
        for ( int k = 0; k < 2; ++k )
        {
            if ( k > 0 && s[1].e == s[0].e )
                continue;
            if ( !s[k].e )
                return FLT_MAX;
            const auto& p = s[1 - k].p;
            const float dSq = ( closestPointOnLineSegm( p, a.edgeSegment( s[k].e ) ) - p ).lengthSq();
            ubSq = std::min( ubSq, std::max( s[k].distSq, dSq ) );
        }
        return ubSq;
    };

    // the most distant point found inside the segments, if it is farther than all vertices
    struct Best
    {
        UndirectedEdgeId be;
        LineCorner c;
    };
    const float startSq = std::max( bv.maxSq, startDistSq );
    // a segment is not halved if none of its points can be farther than this; updated only when a farther point is found
    const float startStopSq = sqr( std::sqrt( startSq ) + tolerance );
    Best start;
    start.c.distSq = startSq;
    Best best = start;
    // zero, negative or NaN tolerance would mean endless subdivision or wrong pruning
    if ( startSq < maxDistanceSq && tolerance > 0 )
    {
        // of several equally distant points, the one in the segment with the smallest id is taken
        best = tbb::parallel_reduce( tbb::blocked_range( 0_ue, UndirectedEdgeId( b.topology.undirectedEdgeSize() ) ), best,
            [&]( const tbb::blocked_range<UndirectedEdgeId>& range, Best init )
            {
                std::vector<Segm> stack;
                for ( UndirectedEdgeId be = range.begin(); be < range.end() && init.c.distSq < maxDistanceSq; ++be )
                {
                    if ( b.topology.isLoneEdge( be ) )
                        continue;
                    // every segment starts from the same distance, so the result does not depend on the order of segments
                    Best cur = start;
                    float stopSq = startStopSq;
                    const Segm root{ bv.corners[b.topology.org( be )], bv.corners[b.topology.dest( be )] };
                    stack.push_back( root );
                    stack.back()[1].t = 1;
                    while ( !stack.empty() )
                    {
                        const auto s = stack.back();
                        stack.pop_back();
                        if ( upperBoundSq( s ) <= stopSq )
                            continue;
                        const float t = 0.5f * ( s[0].t + s[1].t );
                        // not the middle of s[0].p and s[1].p, where rounding errors would accumulate with the depth of subdivision
                        const auto p = ( 1 - t ) * root[0].p + t * root[1].p;
                        if ( p == s[0].p || p == s[1].p )
                            continue; // cannot be halved in floats
                        const auto m = projectCorner( p, t, a, maxDistanceSq );
                        if ( m.distSq > cur.c.distSq )
                        {
                            cur = { be, m };
                            if ( cur.c.distSq >= maxDistanceSq )
                            {
                                stack.clear();
                                break;
                            }
                            stopSq = sqr( std::sqrt( cur.c.distSq ) + tolerance );
                        }
                        stack.push_back( { s[0], m } );
                        stack.push_back( { m, s[1] } );
                    }
                    if ( cur.c.distSq > init.c.distSq )
                        init = cur;
                }
                return init;
            },
            [] ( const Best& x, const Best& y ) { return y.c.distSq > x.c.distSq ? y : x; }
        );
    }

    const auto& bestP = best.be ? best.c.p : bv.corners[bv.maxVert].p;
    if ( hasEdges( a ) )
    {
        const auto proj = findProjectionOnPolyline( bestP, a, maxDistanceSq );
        if ( proj.valid() )
            res.a = a.toEdgePoint( EdgeId( proj.line ), proj.point );
        res.distSq = proj.distSq;
    }
    else
        res.distSq = maxDistanceSq;
    res.b = best.be ? EdgePoint( best.be, best.c.t ) : b.toEdgePoint( bv.maxVert );
    return res;
}

} // anonymous namespace

PolylinePolylineDistanceResult findHausdorffDistanceOneWay( const Polyline3& a, const Polyline3& b, float tolerance, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    MR_TIMER;
    return findMaxDistanceInsideSegments( a, b, tolerance, maxDistanceSq, projectVerts( a, b, rigidB2A, maxDistanceSq ), 0 );
}

PolylinePolylineDistanceResult findHausdorffDistance( const Polyline3& a, const Polyline3& b, float tolerance, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    MR_TIMER;
    std::optional<AffineXf3f> rigidA2B;
    if ( rigidB2A )
        rigidA2B = rigidB2A->inverse();
    const auto bVerts = projectVerts( a, b, rigidB2A, maxDistanceSq );
    const auto aVerts = projectVerts( b, a, rigidA2B ? &*rigidA2B : nullptr, maxDistanceSq );
    // each direction skips the segments, which cannot be farther than the distance already found in the other direction
    auto res = findMaxDistanceInsideSegments( a, b, tolerance, maxDistanceSq, bVerts, aVerts.maxSq );
    const auto ba = findMaxDistanceInsideSegments( b, a, tolerance, maxDistanceSq, aVerts, res.distSq );
    if ( ba.distSq > res.distSq )
        res = { ba.b, ba.a, ba.distSq };
    return res;
}

} //namespace MR
