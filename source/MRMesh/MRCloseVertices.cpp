#include "MRCloseVertices.h"
#include "MRMesh.h"
#include "MRAABBTreePoints.h"
#include "MRPointsInBall.h"
#include "MRParallelFor.h"
#include "MRphmap.h"
#include "MRRingIterator.h"
#include "MRTimer.h"

namespace MR
{

VertMap findSmallestCloseVertices( const VertCoords & points, float closeDist, const VertBitSet * valid )
{
    MR_TIMER

    AABBTreePoints tree( points, valid );
    VertMap res;
    res.resizeNoInit( points.size() );
    ParallelFor( points, [&]( VertId v )
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
    } );
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

    return res;
}

VertMap findSmallestCloseVertices( const Mesh & mesh, float closeDist )
{
    return findSmallestCloseVertices( mesh.points, closeDist, &mesh.topology.getValidVerts() );
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

VertBitSet findCloseVertices( const VertCoords & points, float closeDist, const VertBitSet * valid )
{
    return findCloseVertices( findSmallestCloseVertices( points, closeDist, valid ) );
}

VertBitSet findCloseVertices( const Mesh & mesh, float closeDist )
{
    return findCloseVertices( mesh.points, closeDist, &mesh.topology.getValidVerts() );
}

struct VertPair
{
    VertId a, b;
    friend bool operator ==( const VertPair &, const VertPair & ) = default;
};

} // namespace MR

namespace std
{

template<> 
struct hash<MR::VertPair> 
{
    size_t operator()( MR::VertPair const& p ) const noexcept
    {
        return size_t( p.a ) ^ ( size_t( p.b ) << 16 );
    }
};

} // namespace std

namespace MR
{

EdgeHashMap findTwinEdgeHashMap( const Mesh & mesh, float closeDist )
{
    MR_TIMER
    EdgeHashMap res;

    const auto map = findSmallestCloseVertices( mesh, closeDist );
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
                res[e] = it->second;
                it->second = e;
            }
        }
    }

    return res;
}

EdgeBitSet findTwinEdges( const EdgeHashMap & emap )
{
    MR_TIMER
    EdgeBitSet res;
    for ( const auto & [e1, e2] : emap )
    {
        res.autoResizeSet( e1 );
        res.autoResizeSet( e2 );
    }

    return res;
}

EdgeBitSet findTwinEdges( const Mesh & mesh, float closeDist )
{
    return findTwinEdges( findTwinEdgeHashMap( mesh, closeDist ) );
}

UndirectedEdgeBitSet findTwinUndirectedEdges( const EdgeHashMap & emap )
{
    MR_TIMER
    UndirectedEdgeBitSet res;
    for ( const auto & [e1, e2] : emap )
    {
        res.autoResizeSet( e1.undirected() );
        res.autoResizeSet( e2.undirected() );
    }

    return res;
}

UndirectedEdgeBitSet findTwinUndirectedEdges( const Mesh & mesh, float closeDist )
{
    return findTwinUndirectedEdges( findTwinEdgeHashMap( mesh, closeDist ) );
}

} //namespace MR
