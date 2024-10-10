#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRMeshTopology.h"

MR_EXTERN_C_BEGIN

/// returns closed loop of region boundary starting from given region boundary edge (region faces on the right, and not-region faces or holes on the left);
/// if more than two boundary edges connect in one vertex, then the function makes the most abrupt turn to left
MRMESHC_API MREdgeLoop* mrTrackRightBoundaryLoop( const MRMeshTopology* topology, MREdgeId e0, const MRFaceBitSet* region );

MR_EXTERN_C_END
