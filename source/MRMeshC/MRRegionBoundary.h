#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"

MR_EXTERN_C_BEGIN

MRMESHC_API MREdgeLoop* mrTrackRightBoundaryLoop( const MRMeshTopology* topology, MREdgeId e0, const MRFaceBitSet* region );

MR_EXTERN_C_END