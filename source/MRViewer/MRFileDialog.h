#pragma once
#include "exports.h"
#include "MRMesh/MRIOFilters.h"
#include <filesystem>

namespace MR
{

struct FileParameters
{
    // Default filename
    std::string fileName;
    // Dialog will open this folder for browsing, default: last used folder
    std::filesystem::path baseFolder{};
    IOFilters filters{};
};

// Allow user to select only one file
// returns empty path on cancel
MRVIEWER_API std::filesystem::path openFileDialog( const FileParameters& params = {} );

// Allow user to select several files
// returns empty vector on cancel
MRVIEWER_API std::vector<std::filesystem::path> openFilesDialog( const FileParameters& params = {} );

// Select one folder
// returns empty path on cancel
MRVIEWER_API std::filesystem::path openFolderDialog( std::filesystem::path baseFolder = {} );

// Allow user to select several folders
// returns empty path on cancel
MRVIEWER_API std::vector<std::filesystem::path> openFoldersDialog( std::filesystem::path baseFolder = {} );

// returns empty path on cancel
MRVIEWER_API std::filesystem::path saveFileDialog( const FileParameters& params = {} );
}
