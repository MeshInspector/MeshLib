#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRProgressCallback.h"

namespace MR
{

struct CutAroundEdgePathsParams
{
    /// the mesh is cut along the isolines of surface distance equal to this value around each vertex set
    float distance = 0;

    /// keeps the regions of any two vertex sets from overlapping, even if the sets are closer than (2*distance + minSpacing):
    /// in the vertices where the sum of two distance fields is less than (2*distance + minSpacing),
    /// both values are scaled proportionally to make the sum exactly (2*distance + minSpacing);
    /// so the actual surface gap between two regions is about minSpacing * setsDistance / (2*distance + minSpacing)
    float minSpacing = 0;
};

/// computes surface distance field around each vertex set up to (distance + minSpacing),
/// makes any two fields separated by minSpacing, and cuts the mesh along isolines of each field at given distance;
/// no two vertex sets may share a vertex;
/// \return the region of each vertex set (triangles with distance <= params.distance) in the modified mesh
[[nodiscard]] MRMESH_API Expected<std::vector<FaceBitSet>> cutAroundEdgePaths( Mesh & mesh, const std::vector<VertBitSet> & vertSets,
    const CutAroundEdgePathsParams & params, const ProgressCallback & cb = {} );

} //namespace MR
