#pragma once

#include "MRViewerFwd.h"

namespace MR
{

using FilesLoadedCallback = std::function<void( const std::vector<std::shared_ptr<Object>>& objs, const std::string& errors, const std::string& warnings )>;

struct FileLoadOptions
{
    /// first part of undo name
    const char* undoPrefix = "Open ";

    enum class ReplaceMode
    {
        ContructionBased, ///< replace current scene if new one was loaded from single scene file
        ForceReplace,
        ForceAdd
    };

    /// Determines how to deal with current scene after loading new one
    ReplaceMode replaceMode = ReplaceMode::ContructionBased;

    /// if this callback is set - it is called once when all objects are added to scene
    /// top level objects only are present here
    FilesLoadedCallback loadedCallback;
};

} // namespace MR
