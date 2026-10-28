#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

namespace MR
{

/// computes unitless convexity of the mesh at each vertex on the scale of given radius:
/// for each vertex v considers the edge-connected region of triangles with the centers within given radius from v,
/// finds the best plane for the vertices on the boundary of the region, and returns
/// sum_f( distance( plane, triCenter( f ) ) * dot( plane.n, dirArea( f ) ) ) / radius^3,
/// which approximates the signed volume between the region and the plane:
/// positive in convex areas, negative in concave ones, independent of the sign of plane's normal
[[nodiscard]] MRMESH_API Expected<VertScalars> computeLocalConvexity( const Mesh& mesh, float radius, const ProgressCallback& cb = {} );

/// computes unitless convexity of the point cloud with oriented normals at each point on the scale of given radius:
/// for each point p considers the points q_i within given radius from p,
/// finds the best plane for the points at the distance from p in [radius - shell/2, radius + shell/2], and returns
/// pi * sum_i( distance( plane, q_i ) * dot( plane.n, N_i ) ) / ( sum_i |dot( plane.n, N_i )| * radius ),
/// which approximates the same signed volume divided by radius^3 as the mesh version;
/// \param shell if not positive then the average distance between neighbor points is used
[[nodiscard]] MRMESH_API Expected<VertScalars> computeLocalConvexity( const PointCloud& cloud, float radius, float shell = 0, const ProgressCallback& cb = {} );

} //namespace MR
