#include "MRPolylineProject.h"
#include "MRMesh.h"
#include "MRPolyline.h"
#include "MRAABBTreePolyline.h"
#include "MRLineSegm.h"
#include "MRMatrix2.h"
#include "MRPch/MRTBB.h"
#include <algorithm>
#include <cfloat>

namespace MR
{

template<typename V, typename F>
PolylineProjectionResult<V> findProjectionCore( const V& pt, const AABBTreePolyline<V> & tree, float upDistLimitSq, AffineXf<V>* xf, F && edgeToEndPoints )
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
        SubTask() = default;
        SubTask( typename AABBTreePolyline<V>::NodeId n, float dd ) : n( n ), distSq( dd )
        {
        }
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
        float distSq = ( transformed( tree.nodes()[n].box, xf ).getBoxClosestPointTo( pt ) - pt ).lengthSq();
        return SubTask( n, distSq );
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
            auto proj = closestPointOnLineSegm( pt, { a, b } );

            float distSq = ( proj - pt ).lengthSq();
            if ( distSq < res.distSq )
            {
                res.distSq = distSq;
                res.point = proj;
                res.line = lineId;
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

PolylineProjectionResult2 findProjectionOnPolyline2( const Vector2f& pt, const Polyline2& polyline, float upDistLimitSq, AffineXf2f* xf )
{
    return findProjectionCore( pt, polyline.getAABBTree(), upDistLimitSq, xf, [&]( UndirectedEdgeId ue, Vector2f & a, Vector2f & b ) 
    {
        a = polyline.orgPnt( ue );
        b = polyline.destPnt( ue );
    } );
}

PolylineProjectionResult3 findProjectionOnPolyline( const Vector3f& pt, const Polyline3& polyline, float upDistLimitSq, AffineXf3f* xf )
{
    return findProjectionCore( pt, polyline.getAABBTree(), upDistLimitSq, xf, [&]( UndirectedEdgeId ue, Vector3f & a, Vector3f & b ) 
    {
        a = polyline.orgPnt( ue );
        b = polyline.destPnt( ue );
    } );
}

template<typename V>
PolylineProjectionWithOffsetResult<V> findProjectionOnPolylineWithOffsetT(
    const V& pt, const Polyline<V>& polyline, 
    const Vector<float, UndirectedEdgeId>& offsetPerEdge, /*< offset for each edge of polyline */ 
    float upDistLimit /*= FLT_MAX*/, /*< upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimit and no valid point */ 
    AffineXf<V>* xf /*= nullptr */ )
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
    const Vector<float, UndirectedEdgeId>& offsetPerEdge, float upDistLimit, AffineXf2f* xf )
{
    return findProjectionOnPolylineWithOffsetT( pt, polyline, offsetPerEdge, upDistLimit, xf );
}

PolylineProjectionWithOffsetResult3 findProjectionOnPolylineWithOffset( const Vector3f& pt, const Polyline3& polyline,
    const Vector<float, UndirectedEdgeId>& offsetPerEdge, float upDistLimit, AffineXf3f* xf )
{
    return findProjectionOnPolylineWithOffsetT( pt, polyline, offsetPerEdge, upDistLimit, xf );
}

PolylineProjectionResult3 findProjectionOnMeshEdges( const Vector3f& pt, const Mesh& mesh, const AABBTreePolyline3& tree, float upDistLimitSq, AffineXf3f* xf )
{
    return findProjectionCore( pt, tree, upDistLimitSq, xf, [&]( UndirectedEdgeId ue, Vector3f & a, Vector3f & b ) 
    {
        a = mesh.orgPnt( ue );
        b = mesh.destPnt( ue );
    } );
}

template<typename V>
void findEdgesInBallT( const Polyline<V>& polyline, const V& center, float radius, const FoundEdgeCallback<V>& foundCallback, AffineXf<V>* xf )
{
    if ( !foundCallback )
    {
        assert( false );
        return;
    }

    const auto & tree = polyline.getAABBTree();
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
            auto segm = polyline.edgeSegment( node.leafId() );
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
    findEdgesInBallT( polyline, center, radius, foundCallback, xf );
}

void findEdgesInBall( const Polyline3& polyline, const Vector3f& center, float radius, const FoundEdgeCallback3& foundCallback, AffineXf3f* xf )
{
    findEdgesInBallT( polyline, center, radius, foundCallback, xf );
}

} //namespace MR
