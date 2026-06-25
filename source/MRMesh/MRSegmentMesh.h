#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRFaceFace.h"

namespace MR
{

/// the order of segments construction starting from individual faces;
/// each pair of faces are representatives of two distinct segments to be merged together
using GroupOrder = std::vector<FaceFace>;

/// starting from individual faces, merge them in progressively larger segments until every connected component contains only 1 segment;
/// merge order is guided by the preferences:
/// 1) prefer merging smaller by area segments,
/// 2) prefer merging two segment with long common boundary,
/// 3) prefer merging two segments with low average value of the given curvMetric on the common boundary;
/// take e.g. edgeDihedralAngleMetric() as curvMetric
MRMESH_API Expected<GroupOrder> segmentMesh( const Mesh& mesh,
    const EdgeMetric& curvMetric,
    const ProgressCallback& progress = {} );

/// executes grouping of segments till desired number of segments is reached,
/// then returns the boundary edges in between the segments
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet findSegmentBoundaries( const MeshTopology& topology,
    const GroupOrder& groupOrder, int numSegments,
    FaceColors* outFaceColors = nullptr ); ///< optional output face coloring where all faces of one segment share the same color

} //namespace MR
