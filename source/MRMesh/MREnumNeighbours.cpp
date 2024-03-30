#include "MREnumNeighbours.h"
#include "MRMesh.h"
#include "MRRingIterator.h"
#include "MRTimer.h"
#include <cassert>

namespace MR
{

void EnumNeihbourVertices::run( const MeshTopology & topology, VertId start, const VertPredicate & pred )
{
    MR_TIMER

    assert( start );
    assert( bd_.empty() );
    visited_.resize( topology.vertSize() );

    visited_.set( start );
    bd_.push_back( start );
    while ( !bd_.empty() )
    {
        const auto v = bd_.back();
        bd_.pop_back();
        if ( !pred( v ) )
            continue;
        for ( auto e : orgRing( topology, v ) )
        {
            const auto d = topology.dest( e );
            if ( visited_.test_set( d ) )
                continue;
            bd_.push_back( d );
        }
    }

    visited_.clear();
}

VertScalars computeSpaceDistances( const Mesh& mesh, const PointOnFace & start, float range )
{
    MR_TIMER

    VertScalars res( mesh.topology.vertSize(), FLT_MAX );
    EnumNeihbourVertices e;
    e.run( mesh.topology, mesh.getClosestVertex( start ), [&]( VertId v )
    {
        const auto dist = ( start.point - mesh.points[v] ).length();
        res[v] = dist;
        return dist <= range;
    } );

    return res;
}

VertBitSet findNeighborVerts( const Mesh& mesh, const PointOnFace& start, float range )
{
    MR_TIMER

    VertBitSet res( mesh.topology.vertSize() );
    EnumNeihbourVertices e;
    e.run( mesh.topology, mesh.getClosestVertex( start ), [&] ( VertId v )
    {
        const auto inRange = ( start.point - mesh.points[v] ).length() <= range;
        res.set( v, inRange );
        return inRange; // mb better <= ?
    } );

    return res;
}

void EnumNeihbourFaces::run( const MeshTopology & topology, VertId start, const FacePredicate & pred )
{
    MR_TIMER

    assert( start );
    assert( bd_.empty() );
    visited_.resize( topology.faceSize() );

    for ( auto e : orgRing( topology, start ) )
    {
        auto r = topology.right( e );
        if ( !r )
            continue;
        visited_.set( r );
        bd_.push_back( r );
    }

    while ( !bd_.empty() )
    {
        const auto f = bd_.back();
        bd_.pop_back();
        if ( !pred( f ) )
            continue;
        for ( auto e : leftRing( topology, f ) )
        {
            assert( topology.left( e ) == f );
            auto r = topology.right( e );
            if ( !r )
                continue;
            if ( visited_.test_set( r ) )
                continue;
            bd_.push_back( r );
        }
    }

    visited_.clear();
}

} //namespace MR
