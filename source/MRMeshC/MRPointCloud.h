#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// creates a new PointCloud object
MRMESHC_API MRPointCloud* mrPointCloudNew( const MRVector3f* points, size_t pointsNum );

/// coordinates of points
MRMESHC_API const MRVector3f* mrPointCloudPoints( const MRPointCloud* pc );

MRMESHC_API size_t mrPointCloudPointsNum( const MRPointCloud* pc );

/// unit normal directions of points (can be empty if no normals are known)
MRMESHC_API const MRVector3f* mrPointCloudNormals( const MRPointCloud* pc );

MRMESHC_API size_t mrPointCloudNormalsNum( const MRPointCloud* pc );

/// only points and normals corresponding to set bits here are valid
MRMESHC_API const MRVertBitSet* mrPointCloudValidPoints( const MRPointCloud* pc );

MR_EXTERN_C_END
