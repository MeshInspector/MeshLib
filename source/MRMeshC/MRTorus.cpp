#include "MRTorus.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRTorus.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )

MRMakeTorusParameters mrMakeTorusParametersNew()
{
    return {
        .primaryRadius = 1.0f,
        .secondaryRadius = 0.1f,
        .primaryResolution = 16,
        .secondaryResolution = 16,
        // TODO: points
    };
}

MRMesh* mrMakeTorus( const MRMakeTorusParameters* params )
{
    RETURN_NEW( makeTorus(
        params->primaryRadius,
        params->secondaryRadius,
        params->primaryResolution,
        params->secondaryResolution
        // TODO: points
    ) );
}

MRMesh* mrMakeTorusWithSelfIntersections( const MRMakeTorusParameters* params )
{
    RETURN_NEW( makeTorusWithSelfIntersections(
        params->primaryRadius,
        params->secondaryRadius,
        params->primaryResolution,
        params->secondaryResolution
        // TODO: points
    ) );
}
