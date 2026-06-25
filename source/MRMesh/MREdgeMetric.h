#pragma once

#include "MRMeshFwd.h"
#include "MRPch/MRBindingMacros.h"
#include <functional>

namespace MR
{

/// \ingroup SurfacePathGroup
/// \{

/// metric returning 1 for every edge
[[nodiscard]] MRMESH_API EdgeMetric identityMetric();

/// returns edge's length as a metric;
/// this metric is symmetric: m(e) == m(e.sym())
[[nodiscard]] MRMESH_API EdgeMetric edgeLengthMetric( const Mesh & mesh );
[[nodiscard]] MRMESH_API EdgeMetric edgeLengthMetric( const MeshTopology& topology, const VertCoords& points );

/// returns edge's absolute discrete mean curvature as a metric;
/// the metric is minimal in the planar regions of mesh;
/// this metric is symmetric: m(e) == m(e.sym())
[[nodiscard]] MRMESH_API EdgeMetric discreteAbsMeanCurvatureMetric( const Mesh & mesh );
[[nodiscard]] MRMESH_API EdgeMetric discreteAbsMeanCurvatureMetric( const MeshTopology& topology, const VertCoords& points );

/// returns minus of edge's absolute discrete mean curvature as a metric;
/// the metric is minimal in the most curved regions of mesh;
/// this metric is symmetric: m(e) == m(e.sym())
[[nodiscard]] MRMESH_API EdgeMetric discreteMinusAbsMeanCurvatureMetric( const Mesh & mesh );
[[nodiscard]] MRMESH_API EdgeMetric discreteMinusAbsMeanCurvatureMetric( const MeshTopology& topology, const VertCoords& points );

/// not-negative metric that depends both on edge's length and on the angle between its left and right faces
/// \param angleSinFactor multiplier before dihedral angle sine in edge metric calculation (positive to prefer concave angles, negative - convex)
/// \param angleSinForBoundary consider this dihedral angle sine for boundary edges;
/// this metric is symmetric: m(e) == m(e.sym())
[[nodiscard]] MRMESH_API EdgeMetric edgeCurvMetric( const Mesh & mesh, float angleSinFactor = 2, float angleSinForBoundary = 0 );
[[nodiscard]] MRMESH_API EdgeMetric edgeCurvMetric( const MeshTopology& topology, const VertCoords& points, float angleSinFactor = 2, float angleSinForBoundary = 0 );

/// not-negative metric that depends both on edge's length and on the angle between its left and right faces (ignoring the different between convex and concave)
/// \param angleSinFactor multiplier before absolute value of dihedral angle sine in edge metric calculation: zero - planar case, larger values of (PI - dihedral angle)
/// \param angleSinForBoundary consider this dihedral angle sine for boundary edges;
/// this metric is symmetric: m(e) == m(e.sym())
[[nodiscard]] MRMESH_API EdgeMetric edgeAbsCurvMetric( const Mesh & mesh, float angleSinFactor = 2, float angleSinForBoundary = 0 );
[[nodiscard]] MRMESH_API EdgeMetric edgeAbsCurvMetric( const MeshTopology& topology, const VertCoords& points, float angleSinFactor = 2, float angleSinForBoundary = 0 );

struct DihedralAngleProcessParams
{
    float convexFactor = 1;     ///< positive convex dihedral angles are returned multiplied on this factor
    float concaveFactor = 1;    ///< negative concave dihedral angles are returned multiplied on this factor
    float boundaryValue = 0;    ///< this value will be returned as dihedral angle for boundary edges
};

/// this metric returns edge's dihedral angle computed by MR::dihedralAngle and post-processed based on the given parameters;
/// returned value is NOT multiplied on edge's length as in other metrics;
/// this metric is symmetric: m(e) == m(e.sym())
[[nodiscard]] MRMESH_API EdgeMetric edgeDihedralAngleMetric( const Mesh& mesh, const DihedralAngleProcessParams& params = {} );
[[nodiscard]] MRMESH_API EdgeMetric edgeDihedralAngleMetric( const MeshTopology& topology, const VertCoords& points, const DihedralAngleProcessParams& params = {} );

/// pre-computes the metric for all mesh edges to quickly return it later for any edge;
/// input metric must be symmetric: metric(e) == metric(e.sym())
[[nodiscard]] MRMESH_API EdgeMetric edgeTableSymMetric( const MeshTopology & topology, const EdgeMetric & metric );

/// \}

} // namespace MR
