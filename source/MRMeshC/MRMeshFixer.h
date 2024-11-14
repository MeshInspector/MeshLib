#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN
/// returns all faces that complicate one of mesh holes;
/// hole is complicated if it passes via one vertex more than once;
/// deleting such faces simplifies the holes and makes them easier to fill
MRMESHC_API MRFaceBitSet* mrFindHoleComplicatingFaces( MRMesh* mesh );

MR_EXTERN_C_END
