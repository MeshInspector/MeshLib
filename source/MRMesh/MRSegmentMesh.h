#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRFaceFace.h"

namespace MR
{

using GroupOrder = std::vector<FaceFace>;

MRMESH_API Expected<GroupOrder> segmentMesh( const MeshTopology& topology,
    const EdgeMetric& metric ); ///< sum of this metric to be minimized for the boundaries of segments

[[nodiscard]] MRMESH_API UndirectedEdgeBitSet findSegmentBoundaries( const MeshTopology& topology,
    const GroupOrder& groupOrder, int numSegments );

} //namespace MR
