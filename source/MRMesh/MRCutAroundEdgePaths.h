#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

namespace MR
{

struct CutAroundEdgePathsParams
{
    /// the mesh is cut along the isolines of surface distance equal to this value around each path
    float distance = 0;

    /// the spacing between the regions of any two paths, measured in the adjusted distance fields;
    /// in the vertices where the sum of two distance fields is less than (2*distance + minSpacing),
    /// both values are scaled proportionally to make the sum exactly (2*distance + minSpacing);
    /// so the actual surface gap between two regions is about minSpacing * pathsDistance / (2*distance + minSpacing)
    float minSpacing = 0;
};

/// computes surface distance field around each path up to (distance + minSpacing),
/// makes any two fields separated by minSpacing, and cuts the mesh along isolines of each field at given distance;
/// \return the region of each path (triangles with distance <= params.distance) in the modified mesh
[[nodiscard]] MRMESH_API Expected<std::vector<FaceBitSet>> cutAroundEdgePaths( Mesh & mesh, const std::vector<EdgePath> & paths,
    const CutAroundEdgePathsParams & params );

} //namespace MR
