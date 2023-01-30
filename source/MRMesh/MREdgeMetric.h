#pragma once

#include "MRMeshFwd.h"
#include <functional>

namespace MR
{

/// \defgroup SurfacePathGroup

/// \defgroup EdgePathsGroup Edge Paths
/// \ingroup SurfacePathGroup
/// \{

/// a function that maps an edge to a floating-point value
using EdgeMetric = std::function<float( EdgeId )>;

/// metric returning 1 for every edge
[[nodiscard]] MRMESH_API EdgeMetric identityMetric();

/// returns edge's length as a metric
[[nodiscard]] MRMESH_API EdgeMetric edgeLengthMetric( const Mesh & mesh );

/// returns edge's metric that depends both on edge's length and on the angle between its left and right faces
/// \param angleSinFactor multiplier before dihedral angle sine in edge metric calculation (positive to prefer concave angles, negative - convex)
/// \param angleSinForBoundary consider this dihedral angle sine for boundary edges
[[nodiscard]] MRMESH_API EdgeMetric edgeCurvMetric( const Mesh & mesh, float angleSinFactor = 2, float angleSinForBoundary = 0 );

/// pre-computes the metric for all mesh edges to quickly return it later for any edge
[[nodiscard]] MRMESH_API EdgeMetric edgeTableMetric( const MeshTopology & topology, const EdgeMetric & metric );

/// \}

} // namespace MR
