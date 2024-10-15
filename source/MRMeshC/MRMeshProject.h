#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"
#include "MRMeshPart.h"
#include "MRMeshTriPoint.h"
#include "MRPointOnFace.h"

MR_EXTERN_C_BEGIN

typedef struct MRMeshProjectionResult
{
    /// the closest point on mesh, transformed by xf if it is given
    MRPointOnFace proj;
    /// its barycentric representation
    MRMeshTriPoint mtp;
    /// squared distance from pt to proj
    float distSq;
} MRMeshProjectionResult;

/// optional parameters for \ref mrFindProjection
typedef struct MRFindProjectionParameters
{
    /// upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
    float upDistLimitSq;
    /// mesh-to-point transformation, if not specified then identity transformation is assumed
    const MRAffineXf3f* xf;
    /// low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    float loDistLimitSq;
    // TODO: validFaces
    // TODO: validProjections
} MRFindProjectionParameters;

/// creates a default instance
MRMESHC_API MRFindProjectionParameters mrFindProjectionParametersNew( void );

/// computes the closest point on mesh (or its region) to given point
MRMESHC_API MRMeshProjectionResult mrFindProjection( const MRVector3f* pt, const MRMeshPart* mp, const MRFindProjectionParameters* params );

MR_EXTERN_C_END
