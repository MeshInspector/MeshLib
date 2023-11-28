#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/**
 * \brief Create a band of degenerate faces along the border of the specified region and the rest of the mesh
 * \details The function is useful for extruding the region without changing the existing faces and creating holes
 *
 * @param mesh - the target mesh
 * @param region - the region required to be separated by a band of degenerate faces
 * @param outNewFaces - (optional) output newly generated faces
 */
MRMESH_API void makeDegenerateBandAroundRegion( Mesh& mesh, const FaceBitSet& region, FaceBitSet* outNewFaces = nullptr, UndirectedEdgeBitSet* outExtrudedEdges = nullptr, float* maxEdgeLength = nullptr );

} // namespace MR
