#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"
#include "MRBooleanOperation.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct MRBooleanParameters
{
    MRAffineXf3f* rigidB2A;
    bool mergeAllNonIntersectingComponents;
    MRProgressCallback cb;
} MRBooleanParameters;

typedef struct MRBooleanResult
{
    MRMesh* mesh;
    MRString* errorString;
} MRBooleanResult;

MRBooleanResult mrBoolean( const MRMesh* meshA, const MRMesh* meshB, MRBooleanOperation operation, const MRBooleanParameters* params );

#ifdef __cplusplus
}
#endif
