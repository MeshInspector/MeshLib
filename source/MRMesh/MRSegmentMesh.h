#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRFaceFace.h"

namespace MR
{

/// the order of segments construction starting from individual faces;
/// each pair of faces are representatives of two distinct segments to be merged together
using GroupOrder = std::vector<FaceFace>;

MRMESH_API Expected<GroupOrder> segmentMesh( const Mesh& mesh,
    const EdgeMetric& curvMetric, ///< integral of this metric over segments' boundaries will be maximized, take e.g. edgeDihedralAngleMetric()
    const ProgressCallback& progress = {} );

[[nodiscard]] MRMESH_API UndirectedEdgeBitSet findSegmentBoundaries( const MeshTopology& topology,
    const GroupOrder& groupOrder, int numSegments );

} //namespace MR
