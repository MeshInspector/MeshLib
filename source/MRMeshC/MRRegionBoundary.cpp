#include "MRRegionBoundary.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRRegionBoundary.h"

using namespace MR;

REGISTER_AUTO_CAST( EdgeId )
REGISTER_AUTO_CAST( EdgeLoop )
REGISTER_AUTO_CAST( MeshTopology )

MREdgeLoop* mrTrackRightBoundaryLoop( const MRMeshTopology* topology_, MREdgeId e0_, const MRFaceBitSet* region_ )
{
    ARG( topology ); ARG_VAL( e0 ); ARG_PTR_OF( FaceBitSet, region );
    RETURN_NEW( trackRightBoundaryLoop( topology, e0, region ) );
}
