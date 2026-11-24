#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRFaceFace.h"
#include <cfloat>

namespace MR
{

/// the order of segments construction starting from individual faces;
/// each pair of faces are representatives of two distinct segments to be merged together
using GroupOrder = std::vector<FaceFace>;

/// the value to return from Policy::mergePenalty of MeshSegmenter to forbid merging of two segments
inline constexpr double prohibitMergePenalty = DBL_MAX;

/// starting from individual faces, merge them in progressively larger segments until every connected component contains only 1 segment;
/// merge order is guided by the preferences:
/// 1) prefer merging smaller by area segments,
/// 2) prefer merging two segment with long common boundary,
/// 3) prefer merging two segments with low average value of the given curvMetric on the common boundary;
/// take e.g. edgeDihedralAngleMetric() as curvMetric
MRMESH_API Expected<GroupOrder> segmentMesh( const Mesh& mesh,
    const EdgeMetric& curvMetric,
    const ProgressCallback& progress = {} );

/// segments the mesh using MeshSegmenter with NarrowNeckSegmentPolicy till no more merges are allowed,
/// and returns the faces of the largest by area segment;
/// the parts attached to it only by a narrow neck (e.g. scanning artifacts) are not included in the result
/// \param maxNeckLength two segments are not merged if their common boundary (neck) is shorter than this value,
///        and at least one of them has the boundary with mesh holes longer than the neck
[[nodiscard]] MRMESH_API Expected<FaceBitSet> findLargestNarrowNeckSegment( const Mesh& mesh,
    const EdgeMetric& curvMetric, double maxNeckLength,
    const ProgressCallback& progress = {} );

/// executes grouping of segments till desired number of segments is reached,
/// then returns the boundary edges in between the segments
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet findSegmentBoundaries( const MeshTopology& topology,
    const GroupOrder& groupOrder, int numSegments,
    FaceColors* outFaceColors = nullptr ); ///< optional output face coloring where all faces of one segment share the same color

} //namespace MR
