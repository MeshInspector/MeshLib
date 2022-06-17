#pragma once

#include "MRVector3.h"
#include "MREdgePaths.h"
#include <vector>

namespace MR
{
/**
 * \brief Find the best closed edge loop passing through given edges, "best" is according to given edge metric
 * \ingroup MeshSegmentationGroup
 * 
 * \param includeEdges contain all edges that must be present in the returned loop, probably with reversed direction (should have 2 or 3 elements)
 * \param edgeMetric returned loop will minimize this metric
 * \param dir direction approximately orthogonal to the loop
 */
MRMESH_API std::vector<EdgeId> surroundingContour(
    const Mesh & mesh,
    const std::vector<EdgeId> includeEdges,
    const EdgeMetric & edgeMetric,
    const Vector3f & dir
);

} //namespace MR
