#pragma once

#include "MRMeshFwd.h"
#include "MRUnitInfo.h"
#include <optional>

namespace MR::SceneSave
{

struct Settings
{
    /// units of input coordinates and transformation, to be serialized if the format supports it
    std::optional<LengthUnit> lengthUnit;

    /// to report loading progress and allow the user to cancel it
    ProgressCallback progress;
};

} // namespace MR::ObjectSave

namespace MR::ObjectSave
{

// ObjectSave::Settings and SceneSave::Settings must be distinct classes for
// ObjectSave::ObjectSaver and SceneSave::SceneSaver be distinct types and MR_FORMAT_REGISTRY_DECL does not mix them
struct Settings : public SceneSave::Settings
{
};

} // namespace MR::ObjectSave
