#include "MRSaveSettings.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRSaveSettings.h"

using namespace MR;

REGISTER_AUTO_CAST( SaveSettings )

MRSaveSettings mrSaveSettingsNew( void )
{
    RETURN( SaveSettings() );
}