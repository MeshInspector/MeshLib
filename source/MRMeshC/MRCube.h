#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// creates a mesh representing a cube
/// base is the "lower" corner of the cube coordinates
MRMESHC_API MRMesh* mrMakeCube( const MRVector3f* size, const MRVector3f* base );

MR_EXTERN_C_END
