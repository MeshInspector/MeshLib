#pragma once
#include "MRMeshFwd.h"
#include "MRExpected.h"

namespace MR
{

/// stitches together open twin edges
MRMESH_API Expected<size_t> stitchOpenTwinEdges( Mesh& mesh, float tolerance, const ProgressCallback& cb = {} );

}
