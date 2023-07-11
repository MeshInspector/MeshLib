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
    {
        assert( false );
        return res;
    }

    struct SubTask
    {
        typename AABBTreePolyline<V>::NodeId n;
        float distSq = 0;
    };

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.distSq < res.distSq )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&] ( typename AABBTreePolyline<V>::NodeId n )
    {
        return SubTask{ n, distSqToBox( transformed( tree.nodes()[n].box, xf ) ) };
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
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
        if ( s1.distSq < s2.distSq )
            std::swap( s1, s2 );
        assert( s1.distSq >= s2.distSq );
        addSubTask( s1 ); // larger distance to look later
        addSubTask( s2 ); // smaller distance to look first
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
    [pt]( const Box2f & box ) { return ( box.getBoxClosestPointTo( pt ) - pt ).lengthSq(); },
    [pt]( const LineSegm2f & ls ) { return LineSegm2f{ pt, closestPointOnLineSegm( pt, ls ) }; } );
}

PolylineProjectionResult3 findProjectionOnPolyline( const Vector3f& pt, const Polyline3& polyline, float upDistLimitSq, AffineXf3f* xf, float loDistLimitSq )
{
    return findProjectionCore( polyline.getAABBTree(), upDistLimitSq, xf, [&]( UndirectedEdgeId ue, Vector3f & a, Vector3f & b ) 
    {
        a = polyline.orgPnt( ue );
        b = polyline.destPnt( ue );
    }, loDistLimitSq,
    [pt]( const Box3f & box ) { return ( box.getBoxClosestPointTo( pt ) - pt ).lengthSq(); },
    [pt]( const LineSegm3f & ls ) { return LineSegm3f{ pt, closestPointOnLineSegm( pt, ls ) }; } );
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
    {
        assert( false );
        return res;
    }

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
        typename AABBTreePolyline<V>::NodeId n;
        float dist = 0;
        SubTask() = default;
        SubTask( typename AABBTreePolyline<V>::NodeId n, float d ) : n( n ), dist( d )
        {}
    };

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.dist < res.dist )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&] ( typename AABBTreePolyline<V>::NodeId n )
    {
        float dist = ( ( transformed( tree.nodes()[n].box, xf ).getBoxClosestPointTo( pt ) - pt ).length() - maxOffset );
        return SubTask( n, dist );
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
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
        if ( s1.dist < s2.dist )
            std::swap( s1, s2 );
        assert( s1.dist >= s2.dist );
        addSubTask( s1 ); // larger distance to look later
        addSubTask( s2 ); // smaller distance to look first
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
    [pt]( const Box3f & box ) { return ( box.getBoxClosestPointTo( pt ) - pt ).lengthSq(); },
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
    constexpr int MaxStackSize = 32; // to avoid allocations
    typename AABBTreePolyline<V>::NodeId subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&] ( typename AABBTreePolyline<V>::NodeId n )
    {
        float distSq = ( transformed( tree.nodes()[n].box, xf ).getBoxClosestPointTo( center ) - center ).lengthSq();
        if ( distSq <= radiusSq )
            subtasks[stackSize++] = n;
    };

    addSubTask( tree.rootNodeId() );

    while ( stackSize > 0 )
    {
        const auto n = subtasks[--stackSize];
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
