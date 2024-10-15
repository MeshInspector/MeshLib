#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"

MR_EXTERN_C_BEGIN

typedef struct MRMeshLoadObjLoadSettings
{
    /// if true then vertices will be returned relative to some transformation to avoid precision loss
    bool customXf;
    /// if true, the number of skipped faces (faces than can't be created) will be counted
    bool countSkippedFaces;
    /// callback for set progress and stop process
    MRProgressCallback callback;
}
MRMeshLoadObjLoadSettings;

/// returns a default instance
MRMESHC_API MRMeshLoadObjLoadSettings mrMeshLoadObjLoadSettingsNew( void );

typedef struct MRMeshLoadNamedMesh
{
    const MRString* name;
    const MRMesh* mesh;
    // TODO: uvCoords
    // TODO: colors
    // TODO: textureFiles
    // TODO: texturePerFace
    // TODO: diffuseColor

    /// transform of the loaded mesh, not identity only if ObjLoadSettings.customXf
    MRAffineXf3f xf;
    /// counter of skipped faces (faces than can't be created), not zero only if ObjLoadSettings.countSkippedFaces
    int skippedFaceCount;
    /// counter of duplicated vertices (that created for resolve non-manifold geometry)
    int duplicatedVertexCount;
}
MRMeshLoadNamedMesh;

typedef struct MRVectorMeshLoadNamedMesh MRVectorMeshLoadNamedMesh;

MRMESHC_API const MRMeshLoadNamedMesh mrVectorMeshLoadNamedMeshGet( const MRVectorMeshLoadNamedMesh* vector, size_t index );

MRMESHC_API size_t mrVectorMeshLoadNamedMeshSize( const MRVectorMeshLoadNamedMesh* vector );

MRMESHC_API void mrVectorMeshLoadNamedMeshFree( MRVectorMeshLoadNamedMesh* vector );

/// loads meshes from .obj file
MRMESHC_API MRVectorMeshLoadNamedMesh* mrMeshLoadFromSceneObjFile( const char* file, bool combineAllObjects, const MRMeshLoadObjLoadSettings* settings, MRString** errorString );

MR_EXTERN_C_END
