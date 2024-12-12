#pragma once
#include "MRMeshFwd.h"
#include "MRColor.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN
/// structure with settings and side output parameters for loading point cloud
typedef struct MRPointsLoadSettings
{    
    MRVertColors* colors; ///< points where to load point color map
    MRAffineXf3f* outXf; ///< transform for the loaded point cloud
    MRProgressCallback callback; ///< callback for set progress and stop process
} MRPointsLoadSettings;

MRMESHC_API MRPointsLoadSettings* mrPointsLoadSettingsNew( void );
MRMESHC_API void mrPointsLoadSettingsFree( MRPointsLoadSettings* settings );

MR_EXTERN_C_END