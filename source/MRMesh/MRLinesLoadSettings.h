#pragma once
#include "MRMeshFwd.h"
#include "MRProgressCallback.h"

namespace MR
{

/// setting for polyline loading from external format, and locations of optional output data
struct LinesLoadSettings
{
    VertColors* colors = nullptr;    ///< optional load artifact: per-vertex color map
    ProgressCallback callback;       ///< callback for set progress and stop process
};

} //namespace MR
