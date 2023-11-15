#pragma once

#include "MRObject.h"
#include "MRResumableTask.h"

namespace MR::SceneLoad
{

/// ...
struct SceneLoadResult
{
    /// ...
    std::shared_ptr<Object> scene;
    /// ...
    bool isSceneConstructed = false;
    /// ...
    std::vector<std::filesystem::path> loadedFiles;
    /// ...
    // TODO: user-defined error format
    std::string errorSummary;
    /// ...
    // TODO: user-defined warning format
    std::string warningSummary;
};

/// ...
MRMESH_API SceneLoadResult
fromAnySupportedFormat( const std::vector<std::filesystem::path>& files, ProgressCallback callback = {} );

/// ...
MRMESH_API std::shared_ptr<ResumableTask<SceneLoadResult>>
asyncFromAnySupportedFormat( const std::vector<std::filesystem::path>& files, ProgressCallback callback = {} );

} // namespace MR::SceneLoad
