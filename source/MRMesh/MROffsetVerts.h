#pragma once
#include "MRMeshFwd.h"

namespace MR
{

/// Modifies \p mesh shifting each vertex by the corresponding \p offset
/// @return false if cancelled.
MRMESH_API bool offsetVerts( Mesh& mesh, const VertMetric& offset, ProgressCallback cb );

}
