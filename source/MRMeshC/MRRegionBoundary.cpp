#include "MRRegionBoundary.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRRegionBoundary.h"

using namespace MR;

REGISTER_AUTO_CAST( EdgeId )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( MeshTopology )
REGISTER_VECTOR( EdgeLoop )

MREdgeLoop* mrTrackRightBoundaryLoop( const MRMeshTopology* topology_, MREdgeId e0_, const MRFaceBitSet* region_ )
{
    ARG( topology ); ARG_VAL( e0 ); ARG_PTR( region );
    RETURN_NEW_VECTOR( trackRightBoundaryLoop( topology, e0, region ) );
}
