#pragma once

#include "MRMeshFwd.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

/// parameters for \ref mrFindSignedDistances
typedef struct MRMeshProjectionParameters
{
    /// minimum squared distance from a test point to mesh to be computed precisely,
    /// if a mesh point is found within this distance then it is immediately returned without searching for a closer one
    float loDistLimitSq;

    /// maximum squared distance from a test point to mesh to be computed precisely,
    /// if actual distance is larger than upDistLimit will be returned with not-trusted sign
    float upDistLimitSq;

    /// optional reference mesh to world transformation
    const MRAffineXf3f* refXf;

    /// optional test points to world transformation
    const MRAffineXf3f* xf;
} MRMeshProjectionParameters;

/// initializes a default instance
MRMESHC_API MRMeshProjectionParameters mrMeshProjectionParametersNew( void );

/// Computes signed distances from valid vertices of test mesh to the closest point on the reference mesh:
/// positive value - outside reference mesh, negative - inside reference mesh;
/// this method can return wrong sign if the closest point is located on self-intersecting part of the mesh
MRMESHC_API MRScalars* mrFindSignedDistances( const MRMesh* refMesh, const MRMesh* mesh, const MRMeshProjectionParameters* params );
// TODO: custom projector support

MR_EXTERN_C_END
