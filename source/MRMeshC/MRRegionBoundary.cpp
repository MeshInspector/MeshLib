#include "MRRegionBoundary.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRRegionBoundary.h"
#include "MRMesh/MRBitSet.h"

using namespace MR;

REGISTER_AUTO_CAST( EdgeLoops )
REGISTER_AUTO_CAST( EdgeId )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( VertBitSet )
REGISTER_AUTO_CAST( UndirectedEdgeBitSet )
REGISTER_AUTO_CAST( MeshTopology )
REGISTER_VECTOR( EdgeLoop )

MREdgeLoop* mrTrackRightBoundaryLoop( const MRMeshTopology* topology_, MREdgeId e0_, const MRFaceBitSet* region_ )
{
    ARG( topology ); ARG_VAL( e0 ); ARG_PTR( region );
    RETURN_NEW_VECTOR( trackRightBoundaryLoop( topology, e0, region ) );
}

const MREdgeLoop mrEdgeLoopsGet( const MREdgeLoops* loops_, size_t index )
{
    ARG( loops );
    RETURN_VECTOR( loops[index] );
}

size_t mrEdgeLoopsSize( const MREdgeLoops* loops_ )
{
    ARG( loops );
    return loops.size();
}

void mrEdgeLoopsFree( MREdgeLoops* loops_ )
{
    ARG_PTR( loops );
    delete loops;
}

MREdgeLoops* mrFindRightBoundary( const MRMeshTopology* topology_, const MRFaceBitSet* region_ )
{
    ARG( topology ); ARG_PTR( region );
    RETURN_NEW( findRightBoundary( topology, region ) );
}

MRFaceBitSet* mrGetIncidentFacesFromVerts( const MRMeshTopology* topology_, const MRVertBitSet* region_ )
{
    ARG( topology ); ARG( region );
    RETURN_NEW( getIncidentFaces( topology, region ) );
}

MRFaceBitSet* mrGetIncidentFacesFromEdges( const MRMeshTopology* topology_, const MRUndirectedEdgeBitSet* region_ )
{
    ARG( topology ); ARG( region );
    RETURN_NEW( getIncidentFaces( topology, region ) );
}
