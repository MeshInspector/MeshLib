#pragma once
#include "MRMeshFwd.h"

namespace MR
{

/// Modifies \p mesh shifting each vertex by the corresponding \p offset
MRMESH_API void offsetVerts( Mesh& mesh, const VertMetric& offset, ProgressCallback cb );

}
