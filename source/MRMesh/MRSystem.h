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

// returns path of resource files directory
// .json and .png files
MRMESH_API std::filesystem::path GetResourcesDirectory();

// returns path of font files directory
// .ttf files
MRMESH_API std::filesystem::path GetFontsDirectory();

// returns path of lib files directory
// .dll .so files
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

using FileNamesStack = std::vector<std::filesystem::path>;

} // namespace MR
