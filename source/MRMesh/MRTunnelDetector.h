#pragma once

#include "MREdgeMetric.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"

namespace MR
{

/// \defgroup TunnelDetectorGroup Tunnel Detector
/// \ingroup SurfacePathGroup
/// \{

/// detects all not-contractible-in-point and not-equivalent tunnel loops on the mesh;
/// trying to include in the loops the edges with the smallest metric;
/// if no metric is given then discreteMinusAbsMeanCurvatureMetric will be used
MRMESH_API Expected<std::vector<EdgeLoop>, std::string> detectBasisTunnels( const MeshPart& mp, EdgeMetric metric = {}, ProgressCallback progressCallback = {} );

/// returns tunnels as a number of faces;
/// if you remove these faces and patch every boundary with disk, then the surface will be topology equivalent to sphere
MRMESH_API Expected<FaceBitSet, std::string> detectTunnelFaces( const MeshPart& mp, float maxTunnelLength, EdgeMetric metric = {}, ProgressCallback progressCallback = {} );

/// \}

} // namespace MR
