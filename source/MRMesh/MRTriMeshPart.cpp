#include "MRTriMeshPart.h"
#include "MRTriMesh.h"
#include "MRTimer.h"

namespace MR::detail
{

TriMeshPart::TriMeshPart( const TriMesh & m, const FaceBitSet & r ) : tris( m.tris ), points( m.points ), region( r ), regionVerts( m.points.size() )
{
    MR_TIMER;
    assert( region.find_last() < (int)tris.size() );
    for ( FaceId f : region )
        for ( VertId v : tris[f] )
            regionVerts.set( v );
}

} //namespace MR::detail
