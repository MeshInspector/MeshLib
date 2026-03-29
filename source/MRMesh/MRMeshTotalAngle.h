#pragma once

#include "MRMeshFwd.h"

namespace MR
{

struct FlipRegion
{
    /// Only edges with left and right faces in this set can be flipped
    const FaceBitSet* region = nullptr;

    /// Edges specified by this bit-set will never be flipped
    const UndirectedEdgeBitSet* notFlippable = nullptr;

    /// Only edges with origin or destination in this set before or after flip can be flipped
    const VertBitSet* vertRegion = nullptr;
};

MRMESH_API int reduceTotalAngle( MeshTopology& topology, const VertCoords& points, int numIters, const FlipRegion& region, const ProgressCallback& progressCallback );

} //namespace MR
