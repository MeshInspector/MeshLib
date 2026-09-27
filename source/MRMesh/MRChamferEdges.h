#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRProgressCallback.h"

namespace MR
{

/// replaces given edges of the mesh with chamfer strips: both borders of a strip pass at \param distance from the edges,
/// and the surface in between is cut off; at the corners of the edges the chamfers of neighbor edges meet along the creases;
/// the edges must form one or several disjoint closed loops, not on the mesh boundary, and must be farther than 2*distance apart;
/// the mesh is modified in place, and it can be left partially modified if an error is returned after the input checks
/// \return the triangles of all chamfer strips
[[nodiscard]] MRMESH_API Expected<FaceBitSet> chamferEdges( Mesh & mesh, const UndirectedEdgeBitSet & edges, float distance,
    const ProgressCallback & cb = {} );

} // namespace MR
