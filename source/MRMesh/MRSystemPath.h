#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

#include <filesystem>

namespace MR
{

class SystemPath
{
public:
#ifndef MR_PARSING_FOR_PB11_BINDINGS
    /// get the current executable's file path
    MRMESH_API static Expected<std::filesystem::path> getExecutablePath();

    /// get the MRMesh binary's file path
    MRMESH_API static Expected<std::filesystem::path> getLibraryPath();

    /// get the current executable's directory path
    MRMESH_API static Expected<std::filesystem::path> getExecutableDirectory();

    /// get the MRMesh binary's directory path
    MRMESH_API static Expected<std::filesystem::path> getLibraryDirectory();
#endif

    /// directory category
    enum class Directory
    {
        /// resources (.json, .png)
        Resources,
        /// fonts (.ttf)
        Fonts,
        /// plugins (.dll, .so)
        Plugins,
        /// Python modules (.pyd, .pyi)
        PythonModules,
        Count
    };

    /// get the directory path for specified category
    MRMESH_API static std::filesystem::path getDirectory( Directory dir );
    /// override the directory path for specified category, useful for custom configurations
    MRMESH_API static void overrideDirectory( Directory dir, const std::filesystem::path& path );

    /// get the resource files' directory path
    static std::filesystem::path getResourcesDirectory() { return getDirectory( Directory::Resources ); }
    /// get the font files' directory path
    static std::filesystem::path getFontsDirectory() { return getDirectory( Directory::Fonts ); }
    /// get the plugin binaries' directory path
    static std::filesystem::path getPluginsDirectory() { return getDirectory( Directory::Plugins ); }
    /// get the Python modules' binaries' directory path
    static std::filesystem::path getPythonModulesDirectory() { return getDirectory( Directory::PythonModules ); }

private:
    static SystemPath& instance_();

    std::array<std::filesystem::path, (size_t)Directory::Count> directories_;
};

} // namespace MR
