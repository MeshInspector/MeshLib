#pragma once

#include "MRMeshFwd.h"

namespace MR
{
// holds together settings for makeDegenerateBandAroundRegion
struct MakeDegenerateBandAroundRegionParams
{
    // (optional) output newly generated faces
    FaceBitSet* outNewFaces = nullptr;
    // (optional) output edges orthogonal to the boundary
    UndirectedEdgeBitSet* outExtrudedEdges = nullptr;
    // (optional) return legth of the longest edges from the boundary of the region
    float* maxEdgeLength = nullptr;
    // (optional) map of new vertices to old ones
    VertHashMap* new2OldMap = nullptr;
};

/**
 * \brief Create a band of degenerate faces along the border of the specified region and the rest of the mesh
 * \details The function is useful for extruding the region without changing the existing faces and creating holes
 *
 * @param mesh - the target mesh
 * @param region - the region required to be separated by a band of degenerate faces
 * @param params - optional output parameters
 */
MRMESH_API void makeDegenerateBandAroundRegion( Mesh& mesh, const FaceBitSet& region, const MakeDegenerateBandAroundRegionParams& params = {} );

} // namespace MR
