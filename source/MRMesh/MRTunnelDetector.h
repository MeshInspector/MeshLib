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

/// given not-separating loop on input, finds the loop (also not-separation) that
/// 1) goes from left side of input loop
/// 2) returns to the input loop from its right side
/// 3) goes along the input loop to become closed
/// such that the resulting loop has minimal sum of given metric for its edges;
/// edges 1) and 2) can be only inner or boundary to the given region (they must have region from left or from right)
MRMESH_API Expected<EdgeLoop> findSmallestMetricCoLoop( const MeshTopology& topology, const EdgeLoop& loop, const EdgeMetric& metric,
    const FaceBitSet* region = nullptr );

/// same as \ref findMinimalCoLoop with euclidean edge length metric
MRMESH_API Expected<EdgeLoop> findShortestCoLoop( const MeshPart& mp, const EdgeLoop& loop );

[[nodiscard]] MRMESH_API std::vector<EdgeLoop> findSmallestMetricEquivalentLoops( const MeshTopology& topology, const EdgeLoop& loop, const EdgeMetric& metric,
    const FaceBitSet* region = nullptr );

[[nodiscard]] MRMESH_API std::vector<EdgeLoop> findShortestEquivalentLoops( const MeshPart& mp, const EdgeLoop& loop );

struct DetectTunnelSettings
{
    /// maximal length of tunnel loops to consider
    float maxTunnelLength = FLT_MAX;

    /// maximal number of iterations to detect all tunnels;
    /// on a big mesh with many tunnels even one iteration can take a while
    int maxIters = 1;

    /// metric for detectBasisTunnels,
    /// if no metric is given then discreteMinusAbsMeanCurvatureMetric will be used
    EdgeMetric metric;

    /// if true then for every basis loop, findShortestCoLoop will be called;
    /// it typically results in shorter tunnels found, but requires more time per iteration, and more iterations to find all tunnels
    bool buildCoLoops = true;

    /// if ( buildCoLoops ) then some tunnel loops can be equivalent (e.g. they cut the same handle twice),
    /// this option activates their filtering out, but it is very slow
    bool filterEquivalentCoLoops = false;

    /// to report algorithm progress and cancel from outside
    ProgressCallback progress;
};

/// returns tunnels as a number of faces;
/// if you remove these faces and patch every boundary with disk, then the surface will be topology equivalent to sphere
MRMESH_API Expected<FaceBitSet> detectTunnelFaces( const MeshPart& mp, const DetectTunnelSettings & settings = {} );

/// \}

} // namespace MR
