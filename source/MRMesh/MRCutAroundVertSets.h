#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRProgressCallback.h"

namespace MR
{

struct CutAroundVertSetsParams
{
    /// the mesh is cut along the isolines of surface distance equal to this value around each vertex set
    float distance = 0;

    /// keeps the regions of any two vertex sets from overlapping, even if the sets are closer than (2*distance + gap):
    /// in the vertices where the sum of two distance fields is less than (2*distance + gap),
    /// both values are scaled proportionally to make the sum exactly (2*distance + gap);
    /// so the gap is exact in the scaled fields, and the surface gap between two regions is about gap * setsDistance / (2*distance + gap)
    float gap = 0;
};

/// computes surface distance field around each vertex set up to (distance + gap),
/// scales any two fields to be separated by gap, and cuts the mesh along isolines of each field at given distance;
/// no two vertex sets may share a vertex;
/// \return the region of each vertex set (triangles with distance <= params.distance) in the modified mesh
[[nodiscard]] MRMESH_API Expected<std::vector<FaceBitSet>> cutAroundVertSets( Mesh & mesh, const std::vector<VertBitSet> & vertSets,
    const CutAroundVertSetsParams & params, const ProgressCallback & cb = {} );

} //namespace MR
