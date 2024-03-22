#include "MRExpandShrink.h"
#include "MREdgePaths.h"
#include "MRTimer.h"
#include "MRMeshTopology.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"

namespace MR
{

void expand( const MeshTopology & topology, FaceBitSet & region, int hops )
{
    assert( hops >= 0 );
    if ( hops <= 0 )
        return;
    MR_TIMER
    dilateRegionByMetric( topology, identityMetric(), region, hops + 0.5f );
}

FaceBitSet expand( const MeshTopology & topology, FaceId f, int hops )
{
    MR_TIMER
    FaceBitSet res;
    res.resize( topology.faceSize() );
    res.set( f );
    expand( topology, res, hops );
    return res;
}

void expand( const MeshTopology & topology, VertBitSet & region, int hops )
{
    assert( hops >= 0 );
    if ( hops <= 0 )
        return;
    MR_TIMER
    dilateRegionByMetric( topology, identityMetric(), region, hops + 0.5f );
}

VertBitSet expand( const MeshTopology & topology, VertId v, int hops )
{
    MR_TIMER
    VertBitSet res;
    res.resize( topology.vertSize() );
    res.set( v );
    expand( topology, res, hops );
    return res;
}

void shrink( const MeshTopology & topology, FaceBitSet & region, int hops )
{
    assert( hops >= 0 );
    if ( hops <= 0 )
        return;
    MR_TIMER
    erodeRegionByMetric( topology, identityMetric(), region, hops + 0.5f );
}

void shrink( const MeshTopology & topology, VertBitSet & region, int hops )
{
    assert( hops >= 0 );
    if ( hops <= 0 )
        return;
    MR_TIMER
    region = topology.getValidVerts() - region;
    dilateRegionByMetric( topology, identityMetric(), region, hops + 0.5f );
    region = topology.getValidVerts() - region;
}

FaceBitSet expandFaces( const MeshTopology & topology, const FaceBitSet & region, const UndirectedEdgeBitSet * stopEdges )
{
    MR_TIMER
    FaceBitSet res = topology.getValidFaces();
    BitSetParallelFor( res, [&]( FaceId f )
    {
        if ( region.test( f ) )
            return; // this face was already in the region
        for ( EdgeId e : leftRing( topology, f ) )
            if ( ( !stopEdges || !stopEdges->test( e ) ) && contains( region, topology.right( e ) ) )
                return; // a neighbor face is in the region and the edge in between is not among stop edges
        res.reset( f );
    } );
    return res;
}

FaceBitSet shrinkFaces( const MeshTopology & topology, const FaceBitSet & region, const UndirectedEdgeBitSet * stopEdges )
{
    MR_TIMER
    FaceBitSet res = topology.getValidFaces() & region;
    BitSetParallelFor( res, [&]( FaceId f )
    {
        for ( EdgeId e : leftRing( topology, f ) )
        {
            if ( stopEdges && stopEdges->test( e ) )
                continue; // skip stop edges
            if ( auto r = topology.right( e ); r && !region.test( r ) )
            {
                res.reset( f );
                return; // a neighbor face exists and not in the region
            }
        }
    } );
    return res;
}

FaceBitSet getBoundaryFaces( const MeshTopology & topology, const FaceBitSet & region )
{
    MR_TIMER
    FaceBitSet res = topology.getValidFaces() & region;
    BitSetParallelFor( res, [&]( FaceId f )
    {
        for ( EdgeId e : leftRing( topology, f ) )
        {
            if ( auto r = topology.right( e ); r && !region.test( r ) )
                return; // a neighbor face exists and not in the region, keep true bit
        }
        res.reset( f );
    } );
    return res;
}

} //namespace MR
