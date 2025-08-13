#pragma once

#include "MRFileDialog.h"

namespace MR::detail
{

struct FileDialogParameters : FileParameters
{
    bool folderDialog{false}; // open dialog only
    bool multiselect{true};   // open dialog only
    bool saveDialog{false};   // true for save dialog, false for open
};

/// loads from the configuration the path to last used folder (where the files were last saved or open);
/// returns empty path if no last used folder is set
MRVIEWER_API std::string getLastUsedDir();

/// saves in the configuration the path to last used folder (where the files were last saved or open)
MRVIEWER_API void setLastUsedDir( const std::string& folder );

} // namespace MR::detail
