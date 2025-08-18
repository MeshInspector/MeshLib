#pragma once
#include "exports.h"
#include "MRMesh/MRIOFilters.h"
#include "MRMesh/MRSignal.h"
#include <filesystem>
#include <functional>

namespace MR
{

/// This structure contains global signals for file dialogs, that are called on valid selection of file or folder
struct MRVIEWER_CLASS FileDialogSignals
{
public:
    using SelectFileSignal = Signal<void( const std::filesystem::path& path )>;
    using SelectFilesSignal = Signal<void( const std::vector<std::filesystem::path>& )>;
    using SelectFolderSignal = SelectFileSignal;
    using SelectFoldersSignal = SelectFilesSignal;

    /// returns instance of this holder
    MRVIEWER_API static FileDialogSignals& instance();

    SelectFileSignal onOpenFile; ///< called when one file is selected for opening (`openFileDialog` and `openFileDialogAsync`)
    SelectFilesSignal onOpenFiles; ///< called when several files are selected for opening (`openFilesDialog` and `openFilesDialogAsync`)

    SelectFileSignal onSaveFile; ///< called when file name is selected for saving (`saveFileDialog` and `saveFileDialogAsync`)

    SelectFolderSignal onSelectFolder; ///< called when one folder is selected (we do not now differ reason)(`openFolderDialog` and `openFolderDialogAsync`)
    SelectFoldersSignal onSelectFolders;///< called when several folders are selected (we do not now differ reason)(`openFoldersDialog`)

private:
    FileDialogSignals() = default;
    ~FileDialogSignals() = default;
};


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

// Unified function to select a folder in desktop code and in emscripten
// callback is called inside this function in desktop build and deferred in emscripten build
MRVIEWER_API void openFolderDialogAsync( std::function<void ( const std::filesystem::path& )> callback, std::filesystem::path baseFolder = {} );

// Allow user to select several folders
// returns empty path on cancel
MRVIEWER_API std::vector<std::filesystem::path> openFoldersDialog( std::filesystem::path baseFolder = {} );

// returns empty path on cancel
MRVIEWER_API std::filesystem::path saveFileDialog( const FileParameters& params = {} );

// Unified function to save file in desktop code and in emscripten
// callback is called inside this function in desktop build and deferred in emscripten build
MRVIEWER_API void saveFileDialogAsync( std::function<void( const std::filesystem::path& )> callback, const FileParameters& params = {} );

namespace FileDialog
{

struct Parameters : MR::FileParameters
{
    bool folderDialog{false}; // open dialog only
    bool multiselect{true};   // open dialog only
    bool saveDialog{false};   // true for save dialog, false for open
};

/// loads from the configuration the path (UTF8 encoded) to last used folder (where the files were last saved or open);
/// returns empty path if no last used folder is set
MRVIEWER_API std::string getLastUsedDir();

/// saves in the configuration the path (UTF8 encoded) to last used folder (where the files were last saved or open)
MRVIEWER_API void setLastUsedDir( const std::string& folder );

} // namespace FileDialog

} // namespace MR
