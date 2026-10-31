#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRProgressCallback.h"

namespace MR
{

/// replaces given edges of the mesh with chamfer strips: both borders of a strip pass at \param distance from the edges,
/// and the surface in between is cut off; at the corners of the edges the chamfers of neighbor edges meet along the creases;
/// the edges must form disjoint closed loops or chains with both ends on the mesh boundary, the edges themselves not on the boundary;
/// where other loops or chains, or far parts of the same one, are closer than 2*distance, the chamfer is narrowed not to overlap them;
/// an error is returned if the chamfer does not fit on the two faces around the edges (it would reach another sharp edge or a strong bend);
/// the mesh is modified in place, and it can be left partially modified if an error is returned after the input checks
/// \return the triangles of all chamfer strips
[[nodiscard]] MRMESH_API Expected<FaceBitSet> chamferEdges( Mesh & mesh, const UndirectedEdgeBitSet & edges, float distance,
    const ProgressCallback & cb = {} );

} // namespace MR
