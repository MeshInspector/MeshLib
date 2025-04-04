#pragma once

#include "MRFileDialogInternal.h"

namespace MR::detail
{

/// ...
MRVIEWER_API std::vector<std::filesystem::path> runCocoaFileDialog( const FileDialogParameters& params );

} // namespace MR::detail
