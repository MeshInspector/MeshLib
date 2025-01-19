#include "MRPointsLoadSettings.h"
#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRPointsLoadSettings.h"
#include "MRMesh/MRColor.h"

using namespace MR;

REGISTER_AUTO_CAST( PointsLoadSettings )

MRPointsLoadSettings mrPointsLoadSettingsNew( void )
{
    RETURN( PointsLoadSettings() );
}
