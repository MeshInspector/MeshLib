#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

// TODO: MRMesh* mrMakeConvexHullFromPoints

// computes the mesh of convex hull from given mesh
MRMESHC_API MRMesh* mrMakeConvexHullFromMesh( const MRMesh* mesh );

// computes the mesh of convex hull from given point cloud
MRMESHC_API MRMesh* mrMakeConvexHullFromPointCloud( const MRPointCloud* pointCloud );

MR_EXTERN_C_END
