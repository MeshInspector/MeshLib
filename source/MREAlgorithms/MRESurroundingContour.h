#pragma once
#include "exports.h"
#include <MRMesh/MRMeshFwd.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MREdgePaths.h>
#include <vector>

namespace MRE
{
/**
 * \brief Creating contour passing through given edges in given mesh
 * \ingroup MeshSegmentationGroup
 * 
 * \param includeEdgeOrgs returned contour must pass via all origins of given edges (should have 2 or 3 elements)
 * \param edgeMetric returned counter will minimize this metric
 * \param dir direction approximately orthogonal to the expected contour
 */
MREALGORITHMS_API std::vector<MR::EdgeId> surroundingContour(
    const MR::Mesh & mesh,
    const std::vector<MR::EdgeId> includeEdgeOrgs,
    const MR::EdgeMetric & edgeMetric,
    const MR::Vector3f & dir
);

} //namespace MRE
