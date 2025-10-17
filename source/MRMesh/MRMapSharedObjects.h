#pragma once

#include "MRMeshFwd.h"
#include <unordered_map>
#include <filesystem>

namespace MR
{

/// used during serialization to find objects with shared models and write only one model in each group
struct MapSharedObjects
{
    /// maps Objects to relative Objects ()
    std::unordered_map<const Object*, std::pair<const Object*, int>> map;

    const std::filesystem::path rootFolder;
};

/// used during de-serialization to find objects with shared models (having same link-string)
struct MapLinkToSharedObjectModel
{
    std::unordered_map<std::string, const Object*> map;

    const std::filesystem::path rootFolder;
};

} //namespace MR
