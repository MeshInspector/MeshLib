#pragma once

#include "MRVector3.h"
#include "MREdgePaths.h"
#include <vector>

namespace MR
{

/**
 * \brief Find the best closed edge loop passing through given edges, which minimizes the sum of given edge metric
 * \ingroup MeshSegmentationGroup
 * 
 * \param includeEdges contain all edges that must be present in the returned loop, probably with reversed direction (should have at least 2 elements)
 * \param edgeMetric returned loop will minimize the sum of this metric
 * \param dir direction approximately orthogonal to the loop
 */
[[nodiscard]] MRMESH_API EdgeLoop surroundingContour(
    const Mesh & mesh,
    std::vector<EdgeId> includeEdges,
    const EdgeMetric & edgeMetric,
    const Vector3f & dir
);

/**
 * \brief Find the best closed edge loop passing through given vertices, which minimizes the sum of given edge metric
 * \ingroup MeshSegmentationGroup
 * 
 * \param keyVertices contain all vertices that returned loop must pass (should have at least 2 elements)
 * \param edgeMetric returned loop will minimize the sum of this metric
 * \param dir direction approximately orthogonal to the loop
 */
[[nodiscard]] MRMESH_API EdgeLoop surroundingContour(
    const Mesh & mesh,
    std::vector<VertId> keyVertices,
    const EdgeMetric & edgeMetric,
    const Vector3f & dir
);

} //namespace MR
