#include "MRCylinder.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRCylinder.h"
#include "MRMesh/MRMesh.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )

MRMakeCylinderAdvancedParameters mrMakeCylinderAdvancedParametersNew()
{
    return {
        .radius0 = 0.1f,
        .radius1 = 0.1f,
        .startAngle = 0.0f,
        .arcSize = 2.0f * PI_F,
        .length = 1.0f,
        .resolution = 16,
    };
}

MRMesh* mrMakeCylinderAdvanced( const MRMakeCylinderAdvancedParameters* params )
{
    RETURN_NEW( makeCylinderAdvanced(
        params->radius0,
        params->radius1,
        params->startAngle,
        params->arcSize,
        params->length,
        params->resolution
    ) );
}
