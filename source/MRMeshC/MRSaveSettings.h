#pragma once
#include "MRMeshFwd.h"
#include "MRColor.h"

MR_EXTERN_C_BEGIN

/// determines how to save points/lines/mesh
typedef struct MRSaveSettings
{
    /// true - save valid points/vertices only (pack them);
    /// false - save all points/vertices preserving their indices
    bool onlyValidPoints;
    /// if it is turned on, then higher compression ratios are reached but the order of primitives (triangles in meshes or edges in polylines) is changed;
    /// currently this flag affects the saving in .ctm and .ply formats only
    bool packPrimitives;
    /// optional per-vertex color to save with the geometry
    const MRVertColors* colors;
    /// to report save progress and cancel saving if user desires
    MRProgressCallback progress;
} MRSaveSettings;

MRMESHC_API MRSaveSettings mrSaveSettingsNew( void );

MR_EXTERN_C_END
