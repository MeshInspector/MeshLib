#pragma once

#include "MRMeshFwd.h"

namespace MR
{
// holds together settings for makeDegenerateBandAroundRegion
struct ExtrudeParams
{
    // the region required to be separated by a band of degenerate faces
    const FaceBitSet& region;
    // (optional) output newly generated faces
    FaceBitSet* outNewFaces = nullptr;
    // (optional) output edges orthogonal to the boundary
    UndirectedEdgeBitSet* outExtrudedEdges = nullptr;
    // (optional) return legth of the longest edges from the boundary of the region
    float* maxEdgeLength = nullptr;
};

/**
 * \brief Create a band of degenerate faces along the border of the specified region and the rest of the mesh
 * \details The function is useful for extruding the region without changing the existing faces and creating holes
 *
 * @param mesh - the target mesh
 * @param params - settings including target region and optional output parameters
 */
MRMESH_API void makeDegenerateBandAroundRegion( Mesh& mesh, const ExtrudeParams& params );

} // namespace MR
