#pragma once

#include "MRAffineXf.h"
#include "MRBox.h"
#include "MRMeshFwd.h"
#include "MRId.h"

MR_EXTERN_C_BEGIN

/// creates a new PointCloud object
MRMESHC_API MRPointCloud* mrPointCloudNew( void );

/// creates a new point cloud from existing points
MRMESHC_API MRPointCloud* mrPointCloudFromPoints( const MRVector3f* points, size_t pointsNum );

/// coordinates of points
MRMESHC_API const MRVector3f* mrPointCloudPoints( const MRPointCloud* pc );

MRMESHC_API MRVector3f* mrPointCloudPointsRef( MRPointCloud* pc );

MRMESHC_API size_t mrPointCloudPointsNum( const MRPointCloud* pc );

/// unit normal directions of points (can be empty if no normals are known)
MRMESHC_API const MRVector3f* mrPointCloudNormals( const MRPointCloud* pc );

MRMESHC_API size_t mrPointCloudNormalsNum( const MRPointCloud* pc );

/// only points and normals corresponding to set bits here are valid
MRMESHC_API const MRVertBitSet* mrPointCloudValidPoints( const MRPointCloud* pc );

/// passes through all valid points and finds the minimal bounding box containing all of them;
/// if toWorld transformation is given then returns minimal bounding box in world space
MRMESHC_API MRBox3f mrPointCloudComputeBoundingBox( const MRPointCloud* pc, const MRAffineXf3f* toWorld );

/// appends a point and returns its VertId
MRMESHC_API MRVertId mrPointCloudAddPoint( MRPointCloud* pc, const MRVector3f* point_ );

MRMESHC_API MRVertId mrPointCloudAddPointWithNormal( MRPointCloud* pc, const MRVector3f* point_, const MRVector3f* normal_ );

/// deallocates a PointCloud object
MRMESHC_API void mrPointCloudFree( MRPointCloud* pc );

MR_EXTERN_C_END
