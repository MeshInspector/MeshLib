#pragma once

#include "MRFileDialog.h"

namespace MR::detail
{

/// Open Cocoa file dialog
MRVIEWER_API std::vector<std::filesystem::path> runCocoaFileDialog( const MR::FileDialog::Parameters& params );

} // namespace MR::detail
