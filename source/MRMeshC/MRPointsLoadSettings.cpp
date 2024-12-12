#include "MRPointsLoadSettings.h"
#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRPointsLoadSettings.h"
#include "MRMesh/MRColor.h"

using namespace MR;

REGISTER_AUTO_CAST( PointsLoadSettings )



MRPointsLoadSettings* mrPointsLoadSettingsNew( void )
{
    RETURN_NEW( PointsLoadSettings() );
}

void mrPointsLoadSettingsFree( MRPointsLoadSettings* settings )
{
    if ( !settings )
        return;

    if ( settings->colors )
    { 
        mrVertColorsFree( settings->colors );
        settings->colors = nullptr;
    }
}

