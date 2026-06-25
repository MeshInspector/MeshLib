#pragma once

#include "MRMeshFwd.h"
#include "MRVector2.h"

namespace MR
{

/// Makes a solid-of-revolution mesh. The resulting mesh is symmetrical about the z-axis.
/// The profile points must be in the format { distance to the z-axis; z value }.
MRMESH_API Mesh makeSolidOfRevolution( const Contour2f& profile, int resolution = 16 );

} // namespace MR
