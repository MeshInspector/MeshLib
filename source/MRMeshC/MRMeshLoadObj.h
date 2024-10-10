#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"

MR_EXTERN_C_BEGIN

/// ...
typedef struct MRMeshLoadObjLoadSettings
{
    /// ...
    bool customXf;
    /// ...
    bool countSkippedFaces;
    /// ...
    MRProgressCallback callback;
}
MRMeshLoadObjLoadSettings;

/// ...
MRMESHC_API MRMeshLoadObjLoadSettings mrMeshLoadObjLoadSettingsNew( void );

/// ...
typedef struct MRMeshLoadNamedMesh
{
    const MRString* name;
    const MRMesh* mesh;
    // TODO: uvCoords
    // TODO: colors
    // TODO: textureFiles
    // TODO: texturePerFace
    // TODO: diffuseColor
    /// ...
    MRAffineXf3f xf;
    /// ...
    int skippedFaceCount;
    /// ...
    int duplicatedVertexCount;
}
MRMeshLoadNamedMesh;

/// ...
typedef struct MRVectorMeshLoadNamedMesh MRVectorMeshLoadNamedMesh;

/// ...
MRMESHC_API const MRMeshLoadNamedMesh mrVectorMeshLoadNamedMeshGet( const MRVectorMeshLoadNamedMesh* vector, size_t index );

/// ...
MRMESHC_API size_t mrVectorMeshLoadNamedMeshSize( const MRVectorMeshLoadNamedMesh* vector );

/// ...
MRMESHC_API void mrVectorMeshLoadNamedMeshFree( MRVectorMeshLoadNamedMesh* vector );

/// ...
MRMESHC_API MRVectorMeshLoadNamedMesh* mrMeshLoadFromSceneObjFile( const char* file, bool combineAllObjects, const MRMeshLoadObjLoadSettings* settings, MRString** errorString );

MR_EXTERN_C_END
