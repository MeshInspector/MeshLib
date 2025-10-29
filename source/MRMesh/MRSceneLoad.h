#pragma once

#include "MRObject.h"
#include "MREnums.h"
#include <optional>

namespace MR::SceneLoad
{

/// Scene loading result
struct SceneLoadResult
{
    /// The loaded scene or empty object
    std::shared_ptr<SceneRootObject> scene;
    /// Marks whether the scene was loaded from a single file (false) or was built from scratch (true)
    bool isSceneConstructed = false;
    /// List of successfully loaded files
    std::vector<std::filesystem::path> loadedFiles;
    /// Error summary text
    // TODO: user-defined error format
    std::string errorSummary;
    /// Warning summary text
    // TODO: user-defined warning format
    std::string warningSummary;
};

/// Load scene from file;
/// if both targetUnit and loadedObject.lengthUnit are not nullopt, then adjusts transformations of the loaded objects
MRMESH_API SceneLoadResult fromAnySupportedFormat( const std::vector<std::filesystem::path>& files, const ProgressCallback& callback = {},
    const std::optional<LengthUnit>& targetUnit = {} );

/// Async load scene from file;
/// if both targetUnit and loadedObject.lengthUnit are not nullopt, then adjusts transformations of the loaded objects;
/// calls `postLoadCallback` from a working thread (or from the main thread on single-thread platforms) after all files being loaded
using PostLoadCallback = std::function<void ( SceneLoadResult )>;
MRMESH_API void asyncFromAnySupportedFormat( const std::vector<std::filesystem::path>& files, const PostLoadCallback& postLoadCallback,
    const ProgressCallback& progressCallback = {}, const std::optional<LengthUnit>& targetUnits = {} );

} // namespace MR::SceneLoad
