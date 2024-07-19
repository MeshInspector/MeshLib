#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"

MR_EXTERN_C_BEGIN

/// This class can hold either mesh part or point cloud.
/// It is used for generic algorithms operating with either of them
typedef struct MRMeshOrPoints MRMeshOrPoints;

MRMESHC_API MRMeshOrPoints* mrMeshOrPointsFromMesh( const MRMesh* mesh );

MRMESHC_API MRMeshOrPoints* mrMeshOrPointsFromPointCloud( const MRPointCloud* pc );

/// destructs a MeshOrPoints object
MRMESHC_API void mrMeshOrPointsFree( MRMeshOrPoints* mp );

/// an object and its transformation to global space with other objects
typedef struct MRMeshOrPointsXf MRMeshOrPointsXf;

MRMESHC_API MRMeshOrPointsXf* mrMeshOrPointsXfNew( const MRMeshOrPoints* obj, const MRAffineXf3f* xf );

MRMESHC_API MRMeshOrPointsXf* mrMeshOrPointsXfFromMesh( const MRMesh* mesh, const MRAffineXf3f* xf );

MRMESHC_API MRMeshOrPointsXf* mrMeshOrPointsXfFromPointCloud( const MRPointCloud* pc, const MRAffineXf3f* xf );

/// destructs a MeshOrPointsXf object
MRMESHC_API void mrMeshOrPointsXfFree( MRMeshOrPointsXf* mp );

MR_EXTERN_C_END
