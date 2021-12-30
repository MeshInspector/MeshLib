#include "MRExpandShrink.h"
#include "MREdgePaths.h"
#include "MRTimer.h"
#include "MRMeshTopology.h"

namespace MR
{

void expand( const MeshTopology & topology, FaceBitSet & region, int hops )
{
    assert( hops >= 0 );
    if ( hops <= 0 )
        return;
    dilateRegionByMetric( topology, identityMetric(), region, hops + 0.5f );
}

FaceBitSet expand( const MeshTopology & topology, FaceId f, int hops )
{
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
    dilateRegionByMetric( topology, identityMetric(), region, hops + 0.5f );
}

VertBitSet expand( const MeshTopology & topology, VertId v, int hops )
{
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
    erodeRegionByMetric( topology, identityMetric(), region, hops + 0.5f );
}

void shrink( const MeshTopology & topology, VertBitSet & region, int hops )
{
    assert( hops >= 0 );
    if ( hops <= 0 )
        return;

    region = topology.getValidVerts() - region;
    dilateRegionByMetric( topology, identityMetric(), region, hops + 0.5f );
    region = topology.getValidVerts() - region;
}

} //namespace MR
