#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

namespace MR
{

/// least squares conformal map of a mesh part with disk topology to the plane;
/// two vertices most distant along the main direction of the part are mapped in (0,0) and (1,0)
/// \return the planar coordinates of the vertices of the part, or an error if the part has no non-degenerate triangles
[[nodiscard]] MRMESH_API Expected<VertCoords2> leastSquaresConformalMap( const MeshPart& mp );

} //namespace MR
