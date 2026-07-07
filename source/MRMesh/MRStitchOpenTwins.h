#pragma once
#include "MRMeshFwd.h"

namespace MR
{

/// stitches together open twin edges
MRMESH_API void stitchOpenTwinEdges( Mesh& mesh, float tolerance );

}
