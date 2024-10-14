#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"

MR_EXTERN_C_BEGIN

typedef struct MRMeshSaveNamedXfMesh
{
    const char* name;
    MRAffineXf3f toWorld;
    const MRMesh* mesh;
}
MRMeshSaveNamedXfMesh;

/// saves a number of named meshes in .obj file
// TODO: colors
MRMESHC_API void mrMeshSaveSceneToObj( const MRMeshSaveNamedXfMesh* objects, size_t objectsNum, const char* file, MRString** errorString );

MR_EXTERN_C_END
