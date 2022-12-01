#pragma once
#include "exports.h"
#include "MRMesh/MRIOFilters.h"
#include <filesystem>
#include <functional>

namespace MR
{

struct FileParameters
{
    // Default filename
    std::string fileName;
    // Dialog will open this folder for browsing, default: last used folder
    // this parameter is not used in emscripten build
    std::filesystem::path baseFolder{};
    IOFilters filters{};
};

// Allow user to select only one file
// returns empty path on cancel
MRVIEWER_API std::filesystem::path openFileDialog( const FileParameters& params = {} );

// Unified function to open file in desktop code and in emscripten
// callback is called inside this function in desktop build and deferred in emscripten build
MRVIEWER_API void openFileDialogAsync( std::function<void( const std::filesystem::path& )> callback, const FileParameters& params = {} );

// Allow user to select several files
// returns empty vector on cancel
MRVIEWER_API std::vector<std::filesystem::path> openFilesDialog( const FileParameters& params = {} );

// Unified function to open files in desktop code and in emscripten
// callback is called inside this function in desktop build and deferred in emscripten build
MRVIEWER_API void openFilesDialogAsync( std::function<void( const std::vector<std::filesystem::path>& )> callback, const FileParameters& params = {} );

// Select one folder
// returns empty path on cancel
MRVIEWER_API std::filesystem::path openFolderDialog( std::filesystem::path baseFolder = {} );

// Allow user to select several folders
// returns empty path on cancel
MRVIEWER_API std::vector<std::filesystem::path> openFoldersDialog( std::filesystem::path baseFolder = {} );

// returns empty path on cancel
MRVIEWER_API std::filesystem::path saveFileDialog( const FileParameters& params = {} );

// Unified function to save file in desktop code and in emscripten
// callback is called inside this function in desktop build and deferred in emscripten build
MRVIEWER_API void saveFileDialogAsync( std::function<void( const std::filesystem::path& )> callback, const FileParameters& params = {} );

MRVIEWER_API std::string getCancelMessage( const std::filesystem::path& path );

}
