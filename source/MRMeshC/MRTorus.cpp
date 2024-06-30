#include "MRTorus.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRTorus.h"

using namespace MR;

MRMakeTorusParameters mrMakeTorusParametersDefault()
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
    auto* res = new Mesh( makeTorus(
        params->primaryRadius,
        params->secondaryRadius,
        params->primaryResolution,
        params->secondaryResolution
        // TODO: points
    ) );
    return reinterpret_cast<MRMesh*>( res );
}
