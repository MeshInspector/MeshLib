#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

namespace MR
{

/// least squares conformal map of a mesh part to the plane, meant for a part with disk topology;
/// two vertices most distant along the main direction of the part are mapped in (0,0) and (1,0);
/// the part must be connected via the edges of its non-degenerate triangles: the system is singular in a component
/// without both pinned vertices, which gives an error or meaningless coordinates there
/// \return the planar coordinates of the vertices of the part, or an error for not finite coordinates, no non-degenerate triangles or a degenerate system
[[nodiscard]] MRMESH_API Expected<VertUVCoords> leastSquaresConformalMap( const MeshPart& mp );

} //namespace MR
