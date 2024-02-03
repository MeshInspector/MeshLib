#pragma once
#include "MRMeshFwd.h"
#include "MRColor.h"
#include <filesystem>
#include <string>

namespace MR
{

// sets debug name for the current thread
MRMESH_API void SetCurrentThreadName( const char * name );

// returns path of current exe directory
MRMESH_API std::filesystem::path GetExeDirectory();

/// find and return path to a resource file (.json, .png)
MRMESH_API std::filesystem::path findResourcePath( const std::filesystem::path& path );

/// find and return path to a font file (.ttf)
MRMESH_API std::filesystem::path findFontPath( const std::filesystem::path& path );

/// find and return path to a library file (.dll, .so)
MRMESH_API std::filesystem::path findLibraryPath( const std::filesystem::path& path );

// returns path of resource files directory
// .json and .png files
[[deprecated( "use findResourcePath( filename )" )]]
MRMESH_API std::filesystem::path GetResourcesDirectory();

// returns path of font files directory
// .ttf files
[[deprecated( "use findFontPath( filename )" )]]
MRMESH_API std::filesystem::path GetFontsDirectory();

// returns path of lib files directory
// .dll .so files
[[deprecated( "use findLibraryPath( filename )" )]]
MRMESH_API std::filesystem::path GetLibsDirectory();

// return path to the folder with user config file(s)
MRMESH_API std::filesystem::path getUserConfigDir();

// returns path of config file in APPDATA
MRMESH_API std::filesystem::path getUserConfigFilePath();

// returns temp directory
MRMESH_API std::filesystem::path GetTempDirectory();

// returns home directory
MRMESH_API std::filesystem::path GetHomeDirectory();

// returns data in clipboard
MRMESH_API std::string GetClipboardText();

// sets data in clipboard
MRMESH_API void SetClipboardText( const std::string& text );

#ifdef _WIN32
// returns the folder where Windows installed, typically "C:\Windows"
MRMESH_API std::filesystem::path GetWindowsInstallDirectory();
#endif //_WIN32

// returns version of MR
MRMESH_API std::string GetMRVersionString();

// Opens given link in default browser
MRMESH_API void OpenLink( const std::string& url );

// returns string identification of the CPU
MRMESH_API std::string GetCpuId();

// returns string with OS name with details
MRMESH_API std::string GetDetailedOSName();

// returns string identification of the OS
MRMESH_API std::string getOSNoSpaces();

// sets new handler for operator new if needed for some platforms
MRMESH_API void setNewHandlerIfNeeded();

using FileNamesStack = std::vector<std::filesystem::path>;

} // namespace MR
