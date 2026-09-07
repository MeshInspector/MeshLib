#pragma once
#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRPch/MRBindingMacros.h"
#include <vector>

// MRMesh-internal: the scratch buffers of the sweep-line triangulation's cache, shared with the
// pipelines built around it in this library (the hole-fill plan path). Not part of the public API:
// the references point inside the cache, so they must not outlive it, and the buffers' meaning
// follows the implementation. Ignored by the bindings, which the cache is reachable from anyway.

namespace MR
{

namespace PlanarTriangulation
{

class ISweepLineCache;

/// where a caller composing a pipeline around the triangulation tracks the loops to triangulate,
/// so its per-call locals do not allocate either; the triangulation itself never touches it
MR_BIND_IGNORE MRMESH_API EdgeLoops& sweepCacheLoops( ISweepLineCache& cache );

/// where triangulateDisjointContours*( ..., outPatchMap = nullptr, cache ) leaves the
/// patch->input edge map of the last run
MR_BIND_IGNORE MRMESH_API WholeEdgeMap& sweepCachePatchMap( ISweepLineCache& cache );

/// one slot of the hole-fill-plan peel's polygon scratch (see sweepCachePeelSlots)
struct MR_BIND_IGNORE SweepCachePeelSlot
{
    EdgeId cur;   ///< current polygon edge in the patch, invalid = consumed position
    int refCode;  ///< plan code of cur: not-negative absolute mesh EdgeId, negative - earlier plan edge
    int succ;     ///< next slot around the polygon
};

/// the polygon scratch of the hole-fill plan peel
MR_BIND_IGNORE MRMESH_API std::vector<SweepCachePeelSlot>& sweepCachePeelSlots( ISweepLineCache& cache );

}

}
