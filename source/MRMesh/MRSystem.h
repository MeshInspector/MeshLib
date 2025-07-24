#pragma once
#include "MRPch/MRBindingMacros.h"
#include "MRMeshFwd.h"
#include "MRColor.h"
#include <filesystem>
#include <string>

namespace MR
{

// sets debug name for the current thread
MRMESH_API void SetCurrentThreadName( const char * name );

// returns path of current exe directory
[[deprecated( "Use SystemPath::getExecutableDirectory() instead" )]]
[[nodiscard]] MRMESH_API MR_BIND_IGNORE std::filesystem::path GetExeDirectory();

// returns path of resource files directory
// .json and .png files
[[deprecated( "Use SystemPath::getResourcesDirectory() instead" )]]
[[nodiscard]] MRMESH_API MR_BIND_IGNORE std::filesystem::path GetResourcesDirectory();

// returns path of font files directory
// .ttf files
[[deprecated( "Use SystemPath::getFontsDirectory() instead" )]]
[[nodiscard]] MRMESH_API MR_BIND_IGNORE std::filesystem::path GetFontsDirectory();

// returns path of lib files directory
// .dll .so files
[[deprecated( "Use SystemPath::getPluginsDirectory() instead" )]]
[[nodiscard]] MRMESH_API MR_BIND_IGNORE std::filesystem::path GetLibsDirectory();

// returns path of embedded python modules files directory
// .dll .so files
[[deprecated( "Use SystemPath::getPythonModulesDirectory() instead" )]]
[[nodiscard]] MRMESH_API MR_BIND_IGNORE std::filesystem::path GetEmbeddedPythonDirectory();

// return path to the folder with user config file(s)
[[nodiscard]] MRMESH_API std::filesystem::path getUserConfigDir();

// returns path of config file in APPDATA
[[nodiscard]] MRMESH_API std::filesystem::path getUserConfigFilePath();

// returns temp directory
[[nodiscard]] MRMESH_API std::filesystem::path GetTempDirectory();

// returns home directory
[[nodiscard]] MRMESH_API std::filesystem::path GetHomeDirectory();

#ifdef _WIN32
// returns the folder where Windows installed, typically "C:\Windows"
// This is removed from the bindings because it doesn't exist on all platforms.
[[nodiscard]] MRMESH_API MR_BIND_IGNORE std::filesystem::path GetWindowsInstallDirectory();
#endif //_WIN32

// returns version of MR
[[nodiscard]] MRMESH_API std::string GetMRVersionString();

// Opens given link in default browser
MRMESH_API void OpenLink( const std::string& url );

// Opens given file (or directory) in associated application
MRMESH_API bool OpenDocument( const std::filesystem::path& path );

// returns string identification of the CPU
[[nodiscard]] MRMESH_API std::string GetCpuId();

// returns string with OS name with details
[[nodiscard]] MRMESH_API std::string GetDetailedOSName();

// returns string identification of the OS
[[nodiscard]] MRMESH_API std::string getOSNoSpaces();

// sets new handler for operator new if needed for some platforms
MRMESH_API void setNewHandlerIfNeeded();

using FileNamesStack = std::vector<std::filesystem::path>;

#ifndef __EMSCRIPTEN__
/// returns string representation of the current stacktrace
[[nodiscard]] MRMESH_API std::string getCurrentStacktrace();
#endif

struct SystemMemory
{
    /// total amount of physical memory in the system, in bytes (0 if no info)
    size_t physicalTotal = 0;

    /// available amount of physical memory in the system, in bytes (0 if no info)
    size_t physicalAvailable = 0;
};

/// return information about memory available in the system
[[nodiscard]] MRMESH_API SystemMemory getSystemMemory();

#ifdef _WIN32
// This is removed from the bindings because it's not cross-platform.
struct MR_BIND_IGNORE ProccessMemoryInfo
{
    size_t currVirtual = 0, maxVirtual = 0;
    size_t currPhysical = 0, maxPhysical = 0;
};
[[nodiscard]] MRMESH_API MR_BIND_IGNORE ProccessMemoryInfo getProccessMemoryInfo();
#endif //_WIN32

/// Setups logger:
/// 1) makes stdout sink
/// 2) makes file sink (MRLog.txt)
/// 3) redirect std streams to logger
/// 4) print stacktrace on crash (not in wasm)
/// log level - trace
MRMESH_API void setupLoggerByDefault();

} // namespace MR
