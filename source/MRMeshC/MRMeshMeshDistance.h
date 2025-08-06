#pragma once

#include "MRMeshFwd.h"
#include "MRMeshPart.h"
#include "MRPointOnFace.h"

MR_EXTERN_C_BEGIN

typedef struct MRMeshMeshDistanceResult
{
    /// two closest points: from meshes A and B respectively
    MRPointOnFace a, b;

    /// squared distance between a and b
    float distSq;
} MRMeshMeshDistanceResult;

/**
 * \brief computes minimal distance between two meshes or two mesh regions
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid points
 */
MRMESHC_API MRMeshMeshDistanceResult mrFindDistance( const MRMeshPart* a, const MRMeshPart* b,
    const MRAffineXf3f* rigidB2A, float upDistLimitSq );

MR_EXTERN_C_END
