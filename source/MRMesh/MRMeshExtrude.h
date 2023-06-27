#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// ...
MRMESH_API void makeDegenerateBandAroundRegion( Mesh& mesh, const FaceBitSet& region, FaceBitSet* outNewFaces = nullptr,
                                                FaceMap* old2newMap = nullptr );

} // namespace MR
