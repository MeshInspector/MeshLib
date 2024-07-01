#include "MRCloseVertices.h"
#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRAABBTreePoints.h"
#include "MRPointsInBall.h"
#include "MRParallelFor.h"
#include "MRphmap.h"
#include "MRRingIterator.h"
#include "MRTimer.h"

namespace MR
{

std::optional<VertMap> findSmallestCloseVerticesUsingTree( const VertCoords & points, float closeDist, const AABBTreePoints & tree, const VertBitSet * valid, const ProgressCallback & cb )
{
    MR_TIMER

    VertMap res;
    res.resizeNoInit( points.size() );
    if ( !ParallelFor( points, [&]( VertId v )
    {
        VertId smallestCloseVert = v;
        if ( !valid || valid->test( v ) )
        {
            findPointsInBall( tree, points[v], closeDist, [&]( VertId cv, const Vector3f& )
            {
                if ( cv == v )
                    return;
                smallestCloseVert = std::min( smallestCloseVert, cv );
            } );
        }
        res[v] = smallestCloseVert;
    }, subprogress( cb, 0.0f, 0.8f ) ) )
        return {};

    // after parallel pass, some close vertices can be mapped further

    for ( auto v = 0_v; v < points.size(); ++v )
    {
        if ( valid && !valid->test( v ) )
            continue;
        VertId smallestCloseVert = res[v];
        if ( smallestCloseVert == v )
            continue; // v is the smallest closest by itself
        if ( res[smallestCloseVert] == smallestCloseVert )
            continue; // smallestCloseVert is not mapped further

        // find another closest
        smallestCloseVert = v;
        findPointsInBall( tree, points[v], closeDist, [&]( VertId cv, const Vector3f& )
        {
            if ( cv == v )
                return;
            if ( res[cv] != cv )
                return; // cv vertex is removed by itself
            smallestCloseVert = std::min( smallestCloseVert, cv );
        } );
        res[v] = smallestCloseVert;
    }

    if ( !reportProgress( cb, 1.0f ) )
        return {};
    return res;
}

std::optional<VertMap> findSmallestCloseVertices( const VertCoords & points, float closeDist, const VertBitSet * valid, const ProgressCallback & cb )
{
    MR_TIMER
    AABBTreePoints tree( points, valid );
    return findSmallestCloseVerticesUsingTree( points, closeDist, tree, valid, cb );
}

std::optional<VertMap> findSmallestCloseVertices( const Mesh & mesh, float closeDist, const ProgressCallback & cb )
{
    return findSmallestCloseVerticesUsingTree( mesh.points, closeDist, mesh.getAABBTreePoints(), &mesh.topology.getValidVerts(), cb );
}

std::optional<VertMap> findSmallestCloseVertices( const PointCloud & cloud, float closeDist, const ProgressCallback & cb )
{
    return findSmallestCloseVerticesUsingTree( cloud.points, closeDist, cloud.getAABBTree(), &cloud.validPoints, cb );
}

VertBitSet findCloseVertices( const VertMap & smallestMap )
{
    MR_TIMER
    VertBitSet res;
    for ( auto v = 0_v; v < smallestMap.size(); ++v )
    {
        if ( const auto m = smallestMap[v]; m != v )
        {
            res.autoResizeSet( v );
            assert( m < v );
            res.autoResizeSet( m );
        }
    }
    return res;
}

std::optional<VertBitSet> findCloseVertices( const VertCoords & points, float closeDist, const VertBitSet * valid, const ProgressCallback & cb )
{
    auto x = findSmallestCloseVertices( points, closeDist, valid, cb );
    if ( !x )
        return {};
    return findCloseVertices( *x );
}

std::optional<VertBitSet> findCloseVertices( const Mesh & mesh, float closeDist, const ProgressCallback & cb )
{
    auto x = findSmallestCloseVertices( mesh, closeDist, cb );
    if ( !x )
        return {};
    return findCloseVertices( *x );
}

std::optional<VertBitSet> findCloseVertices( const PointCloud & cloud, float closeDist, const ProgressCallback & cb )
{
    auto x = findSmallestCloseVertices( cloud, closeDist, cb );
    if ( !x )
        return {};
    return findCloseVertices( *x );
}

std::vector<EdgePair> findTwinEdgePairs( const Mesh & mesh, float closeDist )
{
    MR_TIMER
    std::vector<EdgePair> res;

    const auto map = *findSmallestCloseVertices( mesh, closeDist );
    VertBitSet closeVerts = findCloseVertices( map );

    HashMap<VertPair, EdgeId> hmap;
    for ( auto v : closeVerts )
    {
        const auto vm = map[v];
        for ( auto e : orgRing( mesh.topology, v ) )
        {
            assert( mesh.topology.org( e ) == v );
            VertPair vp{ vm, map[mesh.topology.dest( e )] };
            auto [it, inserted] = hmap.insert( { vp, e } );
            if ( !inserted )
            {
                res.push_back( { e, it->second } );
                it->second = e;
            }
        }
    }

    return res;
}

EdgeBitSet findTwinEdges( const std::vector<EdgePair> & pairs )
{
    MR_TIMER
    EdgeBitSet res;
    for ( const auto & [e1, e2] : pairs )
    {
        res.autoResizeSet( e1 );
        res.autoResizeSet( e2 );
    }

    return res;
}

EdgeBitSet findTwinEdges( const Mesh & mesh, float closeDist )
{
    return findTwinEdges( findTwinEdgePairs( mesh, closeDist ) );
}

UndirectedEdgeBitSet findTwinUndirectedEdges( const std::vector<EdgePair> & pairs )
{
    MR_TIMER
    UndirectedEdgeBitSet res;
    for ( const auto & [e1, e2] : pairs )
    {
        res.autoResizeSet( e1.undirected() );
        res.autoResizeSet( e2.undirected() );
    }

    return res;
}

UndirectedEdgeBitSet findTwinUndirectedEdges( const Mesh & mesh, float closeDist )
{
    return findTwinUndirectedEdges( findTwinEdgePairs( mesh, closeDist ) );
}

UndirectedEdgeHashMap findTwinUndirectedEdgeHashMap( const std::vector<EdgePair> & pairs )
{
    MR_TIMER
    UndirectedEdgeHashMap res;
    /// every edge is present twice in (pairs) in both orientations; and
    /// every edge is present twice in (res): once in key and once in value
    res.reserve( pairs.size() );
    auto add = [&]( UndirectedEdgeId u1, UndirectedEdgeId u2 )
    {
        [[maybe_unused]] auto [it, inserted] = res.insert( { u1, u2 } );
        assert( it->second == u2 );
    };
    for ( const auto & [e1, e2] : pairs )
    {
        const auto u1 = e1.undirected();
        const auto u2 = e2.undirected();
        add( u1, u2 );
        add( u2, u1 );
    }
    return res;
}

UndirectedEdgeHashMap findTwinUndirectedEdgeHashMap( const Mesh & mesh, float closeDist )
{
    return findTwinUndirectedEdgeHashMap( findTwinEdgePairs( mesh, closeDist ) );
}

} //namespace MR
