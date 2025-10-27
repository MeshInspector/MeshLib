#pragma once

#include "config.h"
#ifndef MRVIEWER_NO_XDG_DESKTOP_PORTAL
#include "MRFileDialog.h"

namespace MR::detail
{

/// Open XDG Desktop Portal file dialog
MRVIEWER_API std::vector<std::filesystem::path> runPortalFileDialog( const MR::FileDialog::Parameters& params );

} // namespace MR::detail
#endif
