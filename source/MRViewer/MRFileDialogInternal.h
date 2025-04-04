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

MRVIEWER_API std::string getCurrentFolder( const std::filesystem::path& baseFolder = {} );

} // namespace MR::detail
