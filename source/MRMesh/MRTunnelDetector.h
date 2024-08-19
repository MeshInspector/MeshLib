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
