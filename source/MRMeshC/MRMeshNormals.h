#pragma once

#include "MRMeshFwd.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

typedef MRVectorVector3f MRFaceNormals;
typedef MRVectorVector3f MRVertNormals;

/// ...
typedef struct MRMeshNormals
{
    MRFaceNormals* faceNormals;
    MRVertNormals* vertNormals;
} MRMeshNormals;

/// ...
MRMESHC_API MRFaceNormals* mrComputePerFaceNormals( const MRMesh* mesh );

/// ...
MRMESHC_API MRVertNormals* mrComputePerVertNormals( const MRMesh* mesh );

/// ...
MRMESHC_API MRVertNormals* mrComputePerVertPseudoNormals( const MRMesh* mesh );

/// ...
MRMESHC_API MRMeshNormals mrComputeMeshNormals( const MRMesh* mesh );

MR_EXTERN_C_END
