#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// adjusts z-coordinates of (m) vertices to make adjusted (m) similar to (target)
MRMESH_API void replicateZ( Mesh & m, const Mesh & target );

} //namespace MR
