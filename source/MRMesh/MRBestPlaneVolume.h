#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

namespace MR
{

/// for each vertex v considers the edge-connected region of triangles with the centers within given radius from v,
/// finds the best plane for the area-weighted centers of the region's triangles, and returns
/// sum_f( distance( plane, triCenter( f ) ) * dot( plane.n, dirArea( f ) ) ) / radius^3,
/// which approximates the signed volume between the region and the plane:
/// positive in convex areas, negative in concave ones, independent of the sign of plane's normal
[[nodiscard]] MRMESH_API Expected<VertScalars> computeBestPlaneVolumes( const Mesh& mesh, float radius, const ProgressCallback& cb = {} );

} //namespace MR
