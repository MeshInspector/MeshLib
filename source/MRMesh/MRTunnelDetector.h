#pragma once

#include "MREdgeMetric.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include <cfloat>

namespace MR
{

/// \defgroup TunnelDetectorGroup Tunnel Detector
/// \ingroup SurfacePathGroup
/// \{

/// detects all not-contractible-in-point and not-equivalent tunnel loops on the mesh;
/// trying to include in the loops the edges with the smallest metric;
/// if no metric is given then discreteMinusAbsMeanCurvatureMetric will be used
MRMESH_API Expected<std::vector<EdgeLoop>> detectBasisTunnels( const MeshPart& mp, EdgeMetric metric = {}, ProgressCallback progressCallback = {} );

/// given not-trivial loop on input, finds the loop that
/// 1) goes from left side of input loop
/// 2) returns to the input loop from its right side
/// 3) goes along the input loop to become closed
/// such that the resulting loop has minimal sum of given metric for its edges
MRMESH_API Expected<EdgeLoop> findSmallestMetricCoLoop( const MeshTopology& topology, const EdgeLoop& loop, const EdgeMetric& metric );

/// same as \ref findMinimalCoLoop with euclidean edge length metric
MRMESH_API Expected<EdgeLoop> findShortestCoLoop( const Mesh& mesh, const EdgeLoop& loop );

struct DetectTunnelSettings
{
    /// maximal length of tunnel loops to consider
    float maxTunnelLength = FLT_MAX;
    /// maximal number of iterations to detect all tunnels;
    /// on a big mesh with many tunnels even one iteration can take a while
    int maxIters = 1;
    /// if no metric is given then discreteMinusAbsMeanCurvatureMetric will be used
    EdgeMetric metric;
    /// to report algorithm progress and cancel from outside
    ProgressCallback progress;
};

/// returns tunnels as a number of faces;
/// if you remove these faces and patch every boundary with disk, then the surface will be topology equivalent to sphere
MRMESH_API Expected<FaceBitSet> detectTunnelFaces( const MeshPart& mp, const DetectTunnelSettings & settings = {} );

/// \}

} // namespace MR
