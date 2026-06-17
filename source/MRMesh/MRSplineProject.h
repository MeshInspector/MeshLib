#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// every control point of the spline is moved to the closest point on the given mesh;
/// every not-control point of the spline is moved to the closest point in the region between two control points
[[nodiscard]] MRMESH_API Contour3f projectSpline( const Mesh& mesh, const MarkedContour3f& spline );

} //namespace MR
