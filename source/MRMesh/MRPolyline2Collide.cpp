#include "MRPolyline2Collide.h"
#include "MRAABBTreePolyline.h"
#include "MRPolyline.h"
#include "MRPolylineProject.h"
#include "MRTimer.h"
#include "MRMatrix2.h"
#include "MRPch/MRTBB.h"
#include <atomic>

namespace
{
using namespace MR;
bool doSegmentsIntersect( const Vector2d& a0, const Vector2d& a1, const Vector2d& b0, const Vector2d& b1 )
{
    // mb use intersection (LineSegm2f)
    auto avec = a1 - a0;
    if ( cross( avec, b0 - a0 ) * cross( avec, b1 - a0 ) > 0 )
        return false;
    auto bvec = b1 - b0;
    if ( cross( bvec, a0 - b0 ) * cross( bvec, a1 - b0 ) > 0 )
        return false;
    return true;
}

}

namespace MR
{

struct NodeNodePoly
{
    AABBTreePolyline2::NodeId aNode;
    AABBTreePolyline2::NodeId bNode;
    NodeNodePoly( AABBTreePolyline2::NodeId a, AABBTreePolyline2::NodeId b ) : aNode( a ), bNode( b )
    {}
};

std::vector<UndirectedEdgeUndirectedEdge> findCollidingEdges( const Polyline2& a, const Polyline2& b, const AffineXf2f* rigidB2A, bool firstIntersectionOnly )
{
    MR_TIMER;

    std::vector<UndirectedEdgeUndirectedEdge> res;
    const AABBTreePolyline2& aTree = a.getAABBTree();
    const AABBTreePolyline2& bTree = b.getAABBTree();
    if ( aTree.nodes().empty() || bTree.nodes().empty() )
        return res;

    std::vector<NodeNodePoly> subtasks{ { AABBTreePolyline2::NodeId{ 0 }, AABBTreePolyline2::NodeId{ 0 } } };

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.back();
        subtasks.pop_back();

        const auto& aNode = aTree[s.aNode];
        const auto& bNode = bTree[s.bNode];

        const auto overlap = aNode.box.intersection( transformed( bNode.box, rigidB2A ) );
        if ( !overlap.valid() )
            continue;

        if ( aNode.leaf() && bNode.leaf() )
        {
            const auto aUndirEdge = aNode.leafId();
            const auto bUndirEdge = bNode.leafId();
            res.emplace_back( aUndirEdge, bUndirEdge );
            continue;
        }

        if ( !aNode.leaf() && ( bNode.leaf() || aNode.box.volume() >= bNode.box.volume() ) )
        {
            // split aNode
            subtasks.emplace_back( aNode.l, s.bNode );
            subtasks.emplace_back( aNode.r, s.bNode );
        }
        else
        {
            assert( !bNode.leaf() );
            // split bNode
            subtasks.emplace_back( s.aNode, bNode.l );
            subtasks.emplace_back( s.aNode, bNode.r );
        }
    }

    std::atomic<int> firstIntersection{ ( int )res.size() };
    tbb::parallel_for( tbb::blocked_range<int>( 0, ( int )res.size() ),
        [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            int knownIntersection = firstIntersection.load( std::memory_order_relaxed );
            if ( firstIntersectionOnly && knownIntersection < i )
                break;
            Vector2f av[2], bv[2];
            av[0] = a.orgPnt( res[i].aUndirEdge ); av[1] = a.destPnt( res[i].aUndirEdge );
            bv[0] = b.orgPnt( res[i].bUndirEdge ); bv[1] = b.destPnt( res[i].bUndirEdge );
            if ( rigidB2A )
            {
                bv[0] = ( *rigidB2A )( bv[0] );
                bv[1] = ( *rigidB2A )( bv[1] );
            }
            if ( doSegmentsIntersect( Vector2d{ av[0] }, Vector2d{ av[1] }, Vector2d{ bv[0] }, Vector2d{ bv[1] } ) )
            {
                if ( firstIntersectionOnly )
                {
                    while ( knownIntersection > i && !firstIntersection.compare_exchange_strong( knownIntersection, i ) )
                    {
                    }
                    break;
                }
            }
            else
            {
                res[i].aUndirEdge = UndirectedEdgeId{}; //invalidate
            }
        }
    } );

    if ( firstIntersectionOnly )
    {
        int knownIntersection = firstIntersection.load( std::memory_order_relaxed );
        if ( knownIntersection < res.size() )
        {
            res[0] = res[knownIntersection];
            res.erase( res.begin() + 1, res.end() );
        }
        else
            res.clear();
    }
    else
    {
        res.erase( std::remove_if( res.begin(), res.end(), [] ( const UndirectedEdgeUndirectedEdge& uu )
        {
            return !uu.aUndirEdge.valid();
        } ), res.end() );
    }

    return res;
}

std::pair<UndirectedEdgeBitSet, UndirectedEdgeBitSet> findCollidingEdgesBitsets( const Polyline2& a, const Polyline2& b,
    const AffineXf2f* rigidB2A )
{
    const auto pairs = findCollidingEdges( a, b, rigidB2A );
    UndirectedEdgeId aMax, bMax;
    for ( const auto& p : pairs )
    {
        aMax = std::max( aMax, p.aUndirEdge );
        bMax = std::max( bMax, p.bUndirEdge );
    }

    std::pair<UndirectedEdgeBitSet, UndirectedEdgeBitSet> res;
    res.first.resize( aMax + 1 );
    res.second.resize( bMax + 1 );
    for ( const auto& p : pairs )
    {
        res.first.set( p.aUndirEdge );
        res.second.set( p.bUndirEdge );
    }
    return res;
}

inline bool doShareVertex( const VertId av[2], const VertId bv[2] )
{
    for ( int i = 0; i < 2; ++i )
    {
        for ( int j = 0; j < 2; ++j )
        {
            if ( av[i] == bv[j] )
                return true;
        }
    }
    return false;
}

std::vector<UndirectedEdgeUndirectedEdge> findSelfCollidingEdges( const Polyline2& polyline )
{
    MR_TIMER;

    std::vector<UndirectedEdgeUndirectedEdge> res;
    const AABBTreePolyline2& tree = polyline.getAABBTree();
    if ( tree.nodes().empty() )
        return res;

    std::vector<NodeNodePoly> subtasks{ { AABBTreePolyline2::NodeId{ 0 }, AABBTreePolyline2::NodeId{ 0 } } };

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.back();
        subtasks.pop_back();
        const auto& aNode = tree[s.aNode];
        const auto& bNode = tree[s.bNode];

        if ( s.aNode == s.bNode )
        {
            if ( !aNode.leaf() )
            {
                subtasks.emplace_back( aNode.l, aNode.l );
                subtasks.emplace_back( aNode.r, aNode.r );
                subtasks.emplace_back( aNode.l, aNode.r );
            }
            continue;
        }

        const auto overlap = aNode.box.intersection( bNode.box );
        if ( !overlap.valid() )
            continue;

        if ( aNode.leaf() && bNode.leaf() )
        {
            const auto aUndirEdge = aNode.leafId();
            const auto bUndirEdge = bNode.leafId();
            VertId av[2], bv[2];
            av[0] = polyline.topology.org( aUndirEdge ); av[1] = polyline.topology.dest( aUndirEdge );
            bv[0] = polyline.topology.org( bUndirEdge ); bv[1] = polyline.topology.dest( bUndirEdge );
            if ( !doShareVertex( av, bv ) )
                res.emplace_back( aUndirEdge, bUndirEdge );
            continue;
        }

        if ( !aNode.leaf() && ( bNode.leaf() || aNode.box.volume() >= bNode.box.volume() ) )
        {
            // split aNode
            subtasks.emplace_back( aNode.l, s.bNode );
            subtasks.emplace_back( aNode.r, s.bNode );
        }
        else
        {
            assert( !bNode.leaf() );
            // split bNode
            subtasks.emplace_back( s.aNode, bNode.l );
            subtasks.emplace_back( s.aNode, bNode.r );
        }
    }

    tbb::parallel_for( tbb::blocked_range<int>( 0, ( int )res.size() ),
        [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            Vector2f av[2], bv[2];
            av[0] = polyline.orgPnt( res[i].aUndirEdge ); av[1] = polyline.destPnt( res[i].aUndirEdge );
            bv[0] = polyline.orgPnt( res[i].bUndirEdge ); bv[1] = polyline.destPnt( res[i].bUndirEdge );
            if ( !doSegmentsIntersect( Vector2d{ av[0] }, Vector2d{ av[1] }, Vector2d{ bv[0] }, Vector2d{ bv[1] } ) )
                res[i].aUndirEdge = UndirectedEdgeId{}; //invalidate
        }
    } );

    res.erase( std::remove_if( res.begin(), res.end(), [] ( const UndirectedEdgeUndirectedEdge& uu )
    {
        return !uu.aUndirEdge.valid();
    } ), res.end() );

    return res;
}

UndirectedEdgeBitSet findSelfCollidingEdgesBS( const Polyline2& polyline )
{
    auto uus = findSelfCollidingEdges( polyline );
    UndirectedEdgeBitSet res;
    for ( const auto& uu : uus )
    {
        res.autoResizeSet( uu.aUndirEdge );
        res.autoResizeSet( uu.bUndirEdge );
    }
    return res;
}

bool isInside( const Polyline2& a, const Polyline2& b, const AffineXf2f* rigidB2A )
{
    assert( b.topology.isClosed() );

    auto aEdge = a.topology.lastNotLoneEdge();
    if ( !aEdge )
        return true; //consider empty polyline always inside

    auto cols = findCollidingEdges( a, b, rigidB2A );
    if ( !cols.empty() )
        return false; // polyline intersect

    Vector2f aPoint = a.orgPnt( aEdge );
    if ( rigidB2A )
        aPoint = rigidB2A->inverse()( aPoint );

    // if removed then warning C4686: 'MR::findProjectionOnPolyline2': possible change in behavior, change in UDT return calling convention
    static PolylineProjectionResult2 unused;

    auto projRes = findProjectionOnPolyline2( aPoint, b );

    // TODO: this should be separate function (e.g. findSignedProjectionOnPolyline2)
    const EdgeId e = projRes.line;
    const auto& v0 = b.orgPnt( e );
    const auto& v1 = b.destPnt( e );

    auto vecA = v1 - v0;
    auto ray = projRes.point - aPoint;

    return cross( vecA, ray ) > 0.0f;
}

} //namespace MR
