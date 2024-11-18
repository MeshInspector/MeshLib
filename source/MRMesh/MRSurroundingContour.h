#pragma once

#include "MRVector3.h"
#include "MREdgePaths.h"
#include "MRExpected.h"
#include <vector>

namespace MR
{

/**
 * \brief Find the best closed edge loop passing through given edges, which minimizes the sum of given edge metric.
 * The algorithm assumes that input edges can be projected on the plane orthogonal to given direction,
 * then the center point of all input edges is found, and each segment of the searched loop is within infinite pie sector
 * with this center and the borders passing via two sorted input edges.
 * \ingroup MeshSegmentationGroup
 * 
 * \param includeEdges contain all edges in arbitrary order that must be present in the returned loop, probably with reversed direction (should have at least 2 elements)
 * \param edgeMetric returned loop will minimize the sum of this metric
 * \param dir direction approximately orthogonal to the loop
 */
[[nodiscard]] MRMESH_API Expected<EdgeLoop> surroundingContour(
    const Mesh & mesh,
    std::vector<EdgeId> includeEdges,
    const EdgeMetric & edgeMetric,
    const Vector3f & dir
);

/**
 * \brief Find the best closed edge loop passing through given vertices, which minimizes the sum of given edge metric.
 * The algorithm assumes that input vertices can be projected on the plane orthogonal to given direction,
 * then the center point of all input vertices is found, and each segment of the searched loop is within infinite pie sector
 * with this center and the borders passing via two sorted input vertices.
 * \ingroup MeshSegmentationGroup
 * 
 * \param keyVertices contain all vertices in arbitrary order that returned loop must pass (should have at least 2 elements)
 * \param edgeMetric returned loop will minimize the sum of this metric
 * \param dir direction approximately orthogonal to the loop
 */
[[nodiscard]] MRMESH_API Expected<EdgeLoop> surroundingContour(
    const Mesh & mesh,
    std::vector<VertId> keyVertices,
    const EdgeMetric & edgeMetric,
    const Vector3f & dir
);

} //namespace MR
