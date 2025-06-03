#include "MRPolylineProject.h"
#include "MRMesh.h"
#include "MRPolyline.h"
#include "MRAABBTreePolyline.h"
#include "MRLineSegm.h"
#include "MRLine.h"
#include "MRIntersectionPrecomputes.h"
#include "MRRayBoxIntersection.h"
#include "MRIntersection.h"
#include "MRMatrix2.h"
#include "MRMinMaxArg.h"
#include "MRTimer.h"
#include "MRInplaceStack.h"

#include "MRPch/MRTBB.h"

#include <algorithm>
#include <cfloat>

namespace MR
{

template<typename V, typename F, typename B, typename L>
PolylineProjectionResult<V> findProjectionCore( const AABBTreePolyline<V> & tree, float upDistLimitSq, AffineXf<V>* xf,
    F && edgeToEndPoints, float loDistLimitSq, B && distSqToBox, L && closestPointsToLineSegm )
{
    PolylineProjectionResult<V> res;
    res.distSq = upDistLimitSq;
    if ( tree.nodes().empty() )
        return res;

    struct SubTask
    {
        NoInitNodeId n;
        float distSq;
    };
    InplaceStack<SubTask, 32> subtasks;

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.distSq < res.distSq )
            subtasks.push( s );
    };

    auto getSubTask = [&] ( NodeId n )
    {
        return SubTask { n, distSqToBox( transformed( tree.nodes()[n].box, xf ) ) };
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        const auto& node = tree[s.n];
        if ( s.distSq >= res.distSq )
            continue;

        if ( node.leaf() )
        {
            const auto lineId = node.leafId();
            V a, b;
            edgeToEndPoints( lineId, a, b );
            if ( xf )
            {
                a = ( *xf )( a );
                b = ( *xf )( b );
            }

            const auto closest = closestPointsToLineSegm( LineSegm<V>{ a, b } );
            const float distSq = closest.lengthSq();
            if ( distSq < res.distSq )
            {
                res.distSq = distSq;
                res.point = closest.b;
                res.line = lineId;
                if ( distSq <= loDistLimitSq )
                    break;
            }
            continue;
        }

        auto s1 = getSubTask( node.l );
        auto s2 = getSubTask( node.r );
        // add task with smaller distance last to descend there first
        if ( s1.distSq < s2.distSq )
        {
            addSubTask( s2 );
            addSubTask( s1 );
        }
        else
        {
            addSubTask( s1 );
            addSubTask( s2 );
        }
    }

    return res;
}

PolylineProjectionResult2 findProjectionOnPolyline2( const Vector2f& pt, const Polyline2& polyline, float upDistLimitSq, AffineXf2f* xf, float loDistLimitSq )
{
    return findProjectionCore( polyline.getAABBTree(), upDistLimitSq, xf, [&]( UndirectedEdgeId ue, Vector2f & a, Vector2f & b ) 
    {
        a = polyline.orgPnt( ue );
        b = polyline.destPnt( ue );
    }, loDistLimitSq,
    [pt]( const Box2f & box ) { return box.getDistanceSq( pt ); },
    [pt]( const LineSegm2f & ls ) { return LineSegm2f{ pt, closestPointOnLineSegm( pt, ls ) }; } );
}

PolylineProjectionResult3 findProjectionOnPolyline( const Vector3f& pt, const Polyline3& polyline, float upDistLimitSq, AffineXf3f* xf, float loDistLimitSq )
{
    return findProjectionCore( polyline.getAABBTree(), upDistLimitSq, xf, [&]( UndirectedEdgeId ue, Vector3f & a, Vector3f & b ) 
    {
        a = polyline.orgPnt( ue );
        b = polyline.destPnt( ue );
    }, loDistLimitSq,
    [pt]( const Box3f & box ) { return box.getDistanceSq( pt ); },
    [pt]( const LineSegm3f & ls ) { return LineSegm3f{ pt, closestPointOnLineSegm( pt, ls ) }; } );
}

PolylineProjectionResult3Arg findMaxProjectionOnPolyline( const VertCoords& points, const Polyline3& polyline,
    const VertBitSet* pointsRegion, AffineXf3f* xf, float loDistLimitSq )
{
    MR_TIMER;
    std::atomic<float> currMaxDistSq{ loDistLimitSq };
    auto pv = parallel_reduce( tbb::blocked_range( 0_v, points.endId() ), MaxArg<float, VertId>{},
        [&] ( const auto & range, MaxArg<float, VertId> curr )
        {
            for ( VertId v = range.begin(); v < range.end(); ++v )
            {
                if ( !contains( pointsRegion, v ) )
                    continue;
                auto myLoDistLimitSq = currMaxDistSq.load( std::memory_order_relaxed );
                auto myRes = findProjectionOnPolyline( points[v], polyline, FLT_MAX, xf, myLoDistLimitSq );
                while ( myRes.distSq > myLoDistLimitSq && !currMaxDistSq.compare_exchange_strong( myLoDistLimitSq, myRes.distSq, std::memory_order_relaxed ) )
                    {}
                assert( myRes.distSq <= currMaxDistSq );
                curr.include( myRes.distSq, v );
            }
            return curr;
        },
        [] ( MaxArg<float, VertId> a, const MaxArg<float, VertId> & b ) { a.include( b ); return a; }
    );
    PolylineProjectionResult3Arg res;
    if ( pv.arg )
    {
        res.pointId = pv.arg;
        assert( pv.val <= currMaxDistSq ); // it can be less only if the closest distance for all points was smaller than given loDistLimitSq
        static_cast<PolylineProjectionResult3&>( res ) = findProjectionOnPolyline( points[res.pointId], polyline, FLT_MAX, xf, currMaxDistSq );
    }
    return res;
}

PolylineProjectionResult3 findProjectionOnPolyline( const Line3f& ln, const Polyline3& polyline,
    float upDistLimitSq, AffineXf3f* xf, float loDistLimitSq )
{
    return findProjectionCore( polyline.getAABBTree(), upDistLimitSq, xf, [&]( UndirectedEdgeId ue, Vector3f & a, Vector3f & b ) 
    {
        a = polyline.orgPnt( ue );
        b = polyline.destPnt( ue );
    }, loDistLimitSq,
    [ln, prec = IntersectionPrecomputes<float>( ln.d )]( const Box3f & box )
    {
        float distSq = 0;
        float s = -FLT_MAX, e = FLT_MAX;
        if( !rayBoxIntersect( box, RayOrigin<float>{ ln.p }, s, e, prec ) )
            distSq = closestPoints( ln, box ).lengthSq();
        return distSq;
    },
    [ln]( const LineSegm3f & ls ) { return closestPoints( ln, ls ); } );
}

template<typename V>
PolylineProjectionWithOffsetResult<V> findProjectionOnPolylineWithOffsetT(
    const V& pt, const Polyline<V>& polyline, 
    const Vector<float, UndirectedEdgeId>& offsetPerEdge,
    float upDistLimit,
    AffineXf<V>* xf,
    float loDistLimit )
{
    const AABBTreePolyline<V>& tree = polyline.getAABBTree();

    PolylineProjectionWithOffsetResult<V> res;
    res.dist = upDistLimit;
    if ( tree.nodes().empty() )
        return res;

    if ( offsetPerEdge.size() < polyline.topology.undirectedEdgeSize() )
    {
        assert( false );
        return res;
    }

    float maxOffset = tbb::parallel_reduce
    (
        tbb::blocked_range( 0_ue, offsetPerEdge.endId() ),
        -FLT_MAX, 
        [&] ( const tbb::blocked_range<UndirectedEdgeId>& range, float max )
        {
            for ( auto i = range.begin(); i < range.end(); ++i )
                max = std::max( max, offsetPerEdge[i] );
            return max;
        }, 
        [] ( float a, float b ) -> float
        {
            return a > b ? a : b;
        }
    );

    struct SubTask
    {
        NoInitNodeId n;
        float dist;
    };
    InplaceStack<SubTask, 32> subtasks;

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.dist < res.dist )
            subtasks.push( s );
    };

    auto getSubTask = [&] ( NodeId n )
    {
        const auto & box = tree.nodes()[n].box;
        float dist = std::sqrt( xf ? transformed( box, *xf ).getDistanceSq( pt ) : box.getDistanceSq( pt ) ) - maxOffset;
        return SubTask { n, dist };
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        const auto& node = tree[s.n];
        if ( s.dist >= res.dist )
            continue;

        if ( node.leaf() )
        {
            const auto lineId = node.leafId();
            V a, b;
            a = polyline.orgPnt( lineId );
            b = polyline.destPnt( lineId );
            if ( xf )
            {
                a = ( *xf )( a );
                b = ( *xf )( b );
            }
            auto proj = closestPointOnLineSegm( pt, { a, b } );

            float dist = ( proj - pt ).length() - offsetPerEdge[lineId];
            if ( dist < res.dist )
            {
                res.dist = dist;
                res.point = proj;
                res.line = lineId;
                if ( dist <= loDistLimit )
                    break;
            }
            continue;
        }

        auto s1 = getSubTask( node.l );
        auto s2 = getSubTask( node.r );
        // add task with smaller distance last to descend there first
        if ( s1.dist < s2.dist )
        {
            addSubTask( s2 );
            addSubTask( s1 );
        }
        else
        {
            addSubTask( s1 );
            addSubTask( s2 );
        }
    }

    return res;
}

Polyline2ProjectionWithOffsetResult findProjectionOnPolyline2WithOffset( const Vector2f& pt, const Polyline2& polyline,
    const Vector<float, UndirectedEdgeId>& offsetPerEdge, float upDistLimit, AffineXf2f* xf, float loDistLimit )
{
    return findProjectionOnPolylineWithOffsetT( pt, polyline, offsetPerEdge, upDistLimit, xf, loDistLimit );
}

PolylineProjectionWithOffsetResult3 findProjectionOnPolylineWithOffset( const Vector3f& pt, const Polyline3& polyline,
    const Vector<float, UndirectedEdgeId>& offsetPerEdge, float upDistLimit, AffineXf3f* xf, float loDistLimit )
{
    return findProjectionOnPolylineWithOffsetT( pt, polyline, offsetPerEdge, upDistLimit, xf, loDistLimit );
}

PolylineProjectionResult3 findProjectionOnMeshEdges( const Vector3f& pt, const Mesh& mesh, const AABBTreePolyline3& tree, float upDistLimitSq, AffineXf3f* xf, float loDistLimitSq )
{
    return findProjectionCore( tree, upDistLimitSq, xf, [&]( UndirectedEdgeId ue, Vector3f & a, Vector3f & b ) 
    {
        a = mesh.orgPnt( ue );
        b = mesh.destPnt( ue );
    }, loDistLimitSq,
    [pt]( const Box3f & box ) { return box.getDistanceSq( pt ); },
    [pt]( const LineSegm3f & ls ) { return LineSegm3f{ pt, closestPointOnLineSegm( pt, ls ) }; } );
}

PolylineProjectionResult3 findProjectionOnMeshEdges( const Line3f& ln, const Mesh& mesh, const AABBTreePolyline3& tree, float upDistLimitSq, AffineXf3f* xf, float loDistLimitSq )
{
    return findProjectionCore( tree, upDistLimitSq, xf, [&]( UndirectedEdgeId ue, Vector3f & a, Vector3f & b ) 
    {
        a = mesh.orgPnt( ue );
        b = mesh.destPnt( ue );
    }, loDistLimitSq,
    [ln, prec = IntersectionPrecomputes<float>( ln.d )]( const Box3f & box )
    {
        float distSq = 0;
        float s = -FLT_MAX, e = FLT_MAX;
        if( !rayBoxIntersect( box, RayOrigin<float>{ ln.p }, s, e, prec ) )
            distSq = closestPoints( ln, box ).lengthSq();
        return distSq;
    },
    [ln]( const LineSegm3f & ls ) { return closestPoints( ln, ls ); } );
}

template<typename V, typename F>
void findEdgesInBallCore( const AABBTreePolyline<V>& tree, const V& center, 
    float radius, const FoundEdgeCallback<V>& foundCallback, AffineXf<V>* xf, F&& edgeToEndPoints )
{
    if ( !foundCallback )
    {
        assert( false );
        return;
    }

    if ( tree.nodes().empty() )
        return;

    const auto radiusSq = sqr( radius );
    InplaceStack<NoInitNodeId, 32> subtasks;

    auto addSubTask = [&] ( NodeId n )
    {
        const auto & box = tree.nodes()[n].box;
        float distSq = xf ? transformed( box, *xf ).getDistanceSq( center ) : box.getDistanceSq( center );
        if ( distSq <= radiusSq )
            subtasks.push( n );
    };

    addSubTask( tree.rootNodeId() );

    while ( !subtasks.empty() )
    {
        const auto n = subtasks.top();
        subtasks.pop();
        const auto& node = tree[n];

        if ( node.leaf() )
        {
            LineSegm<V> segm;
            edgeToEndPoints( node.leafId(), segm.a, segm.b );
            if ( xf )
            {
                segm.a = ( *xf )( segm.a );
                segm.b = ( *xf )( segm.b );
            }
            auto proj = closestPointOnLineSegm( center, segm );

            float distSq = ( proj - center ).lengthSq();
            if ( distSq <= radiusSq )
                foundCallback( node.leafId(), proj, distSq );
            continue;
        }

        addSubTask( node.r ); // look at right node later
        addSubTask( node.l ); // look at left node first
    }
}

void findEdgesInBall( const Polyline2& polyline, const Vector2f& center, float radius, const FoundEdgeCallback2& foundCallback, AffineXf2f* xf )
{
    findEdgesInBallCore( polyline.getAABBTree(), center, radius, foundCallback, xf, [&] ( UndirectedEdgeId ue, Vector2f& a, Vector2f& b )
    {
        a = polyline.orgPnt( ue );
        b = polyline.destPnt( ue );
    } );
}

void findEdgesInBall( const Polyline3& polyline, const Vector3f& center, float radius, const FoundEdgeCallback3& foundCallback, AffineXf3f* xf )
{
    findEdgesInBallCore( polyline.getAABBTree(), center, radius, foundCallback, xf, [&] ( UndirectedEdgeId ue, Vector3f& a, Vector3f& b )
    {
        a = polyline.orgPnt( ue );
        b = polyline.destPnt( ue );
    } );
}

void findMeshEdgesInBall( const Mesh& mesh, const AABBTreePolyline3& tree, const Vector3f& center, float radius, const FoundEdgeCallback3& foundCallback, AffineXf3f* xf /*= nullptr */ )
{
    findEdgesInBallCore( tree, center, radius, foundCallback, xf, [&] ( UndirectedEdgeId ue, Vector3f& a, Vector3f& b )
    {
        a = mesh.orgPnt( ue );
        b = mesh.destPnt( ue );
    } );
}

} //namespace MR
