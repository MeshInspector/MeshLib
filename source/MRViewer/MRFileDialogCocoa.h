#pragma once

#include "MRFileDialogInternal.h"

namespace MR::detail
{

/// Open Cocoa file dialog
MRVIEWER_API std::vector<std::filesystem::path> runCocoaFileDialog( const FileDialogParameters& params );

} // namespace MR::detail
