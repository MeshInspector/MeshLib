#include "MRRegionBoundary.h"

#include "MRMesh/MRRegionBoundary.h"

using namespace MR;

MREdgeLoop* mrTrackRightBoundaryLoop( const MRMeshTopology* topology_, MREdgeId e0_, const MRFaceBitSet* region_ )
{
    const auto& topology = *reinterpret_cast<const MeshTopology*>( topology_ );
    auto e0 = *reinterpret_cast<EdgeId*>( &e0_ );
    const auto* region = reinterpret_cast<const FaceBitSet*>( region_ );

    auto res = trackLeftBoundaryLoop( topology, e0, region );
    return reinterpret_cast<MREdgeLoop*>( new EdgeLoop( std::move( res ) ) );
}
