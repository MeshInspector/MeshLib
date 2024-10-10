#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"
#include "MRMeshPart.h"
#include "MRMeshTriPoint.h"
#include "MRPointOnFace.h"

MR_EXTERN_C_BEGIN

/// ...
typedef struct MRMeshProjectionResult
{
    /// ...
    MRPointOnFace proj;
    /// ...
    MRMeshTriPoint mtp;
    /// ...
    float distSq;
} MRMeshProjectionResult;

/// ...
typedef struct MRFindProjectionParameters
{
    /// ...
    float upDistLimitSq;
    /// ...
    const MRAffineXf3f* xf;
    /// ...
    float loDistLimitSq;
    // TODO: validFaces
    // TODO: validProjections
} MRFindProjectionParameters;

/// ...
MRMESHC_API MRFindProjectionParameters mrFindProjectionParametersNew( void );

/// ...
MRMESHC_API MRMeshProjectionResult mrFindProjection( const MRVector3f* pt, const MRMeshPart* mp, const MRFindProjectionParameters* params );

MR_EXTERN_C_END
