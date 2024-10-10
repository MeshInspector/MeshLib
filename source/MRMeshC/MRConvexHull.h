#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

// TODO: MRMesh* mrMakeConvexHullFromPoints

/// ...
MRMESHC_API MRMesh* mrMakeConvexHullFromMesh( const MRMesh* mesh );

/// ...
MRMESHC_API MRMesh* mrMakeConvexHullFromPointCloud( const MRPointCloud* pointCloud );

MR_EXTERN_C_END
