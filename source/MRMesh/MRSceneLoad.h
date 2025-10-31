#pragma once

#include "MRObject.h"
#include "MRUnitInfo.h"
#include <optional>

namespace MR::SceneLoad
{

struct Settings
{
    /// if both targetUnit and loadedObject.lengthUnit are not nullopt,
    /// then adjusts transformations of the loaded objects to match target units
    std::optional<LengthUnit> targetUnit;

    /// to report loading progress and allow the user to cancel it
    ProgressCallback progress;
};

/// Scene loading result
struct Result
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
MRMESH_API Result fromAnySupportedFormat( const std::vector<std::filesystem::path>& files, const Settings& settings = {} );

/// Async load scene from file;
/// calls `postLoadCallback` from a working thread (or from the main thread on single-thread platforms) after all files being loaded
using PostLoadCallback = std::function<void ( Result )>;
MRMESH_API void asyncFromAnySupportedFormat( const std::vector<std::filesystem::path>& files, const PostLoadCallback& postLoadCallback, const Settings& settings = {} );

} // namespace MR::SceneLoad
