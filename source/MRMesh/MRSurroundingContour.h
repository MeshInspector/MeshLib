#pragma once

#include "MRVector3.h"
#include "MREdgePaths.h"
#include <vector>

namespace MR
{
/**
 * \brief Creating contour passing through given edges in given mesh
 * \ingroup MeshSegmentationGroup
 * 
 * \param includeEdgeOrgs returned contour must pass via all origins of given edges (should have 2 or 3 elements)
 * \param edgeMetric returned counter will minimize this metric
 * \param dir direction approximately orthogonal to the expected contour
 */
MRMESH_API std::vector<EdgeId> surroundingContour(
    const Mesh & mesh,
    const std::vector<EdgeId> includeEdgeOrgs,
    const EdgeMetric & edgeMetric,
    const Vector3f & dir
);

} //namespace MR
