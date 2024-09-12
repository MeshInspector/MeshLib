#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

#include <filesystem>

namespace MR
{

class SystemPath
{
public:
    /// ...
    MRMESH_API static Expected<std::filesystem::path> getExecutablePath();

    /// ...
    MRMESH_API static Expected<std::filesystem::path> getLibraryPath();

    /// ...
    MRMESH_API static Expected<std::filesystem::path> getExecutableDirectory();

    /// ...
    MRMESH_API static Expected<std::filesystem::path> getLibraryDirectory();

    enum class Directory
    {
        Resources,
        Fonts,
        Plugins,
        PythonModules,
        Count
    };

    /// ...
    MRMESH_API static std::filesystem::path getDirectory( Directory dir );
    /// ...
    MRMESH_API static void overrideDirectory( Directory dir, const std::filesystem::path& path );

    /// ...
    static std::filesystem::path getResourcesDirectory() { return getDirectory( Directory::Resources ); }
    /// ...
    static std::filesystem::path getFontsDirectory() { return getDirectory( Directory::Fonts ); }
    /// ...
    static std::filesystem::path getPluginsDirectory() { return getDirectory( Directory::Plugins ); }
    /// ...
    static std::filesystem::path getPythonModulesDirectory() { return getDirectory( Directory::PythonModules ); }

private:
    static SystemPath& instance_();

    std::array<std::filesystem::path, (size_t)Directory::Count> directories_;
};

} // namespace MR
