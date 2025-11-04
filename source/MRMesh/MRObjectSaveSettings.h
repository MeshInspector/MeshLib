#pragma once

#include "MRMeshFwd.h"
#include "MRUnitInfo.h"
#include <optional>

namespace MR::ObjectSave
{

struct Settings
{
    /// units of input coordinates and transformation, to be serialized if the format supports it
    std::optional<LengthUnit> lengthUnit;

    /// to report loading progress and allow the user to cancel it
    ProgressCallback progress;
};

} // namespace MR::ObjectSave
