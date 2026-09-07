#pragma once

#include "config.h"
#if !defined( _WIN32 ) && !defined( MRVIEWER_NO_XDG_DESKTOP_PORTAL )
#include "MRFileDialog.h"

namespace MR::detail
{

/// Checks if XDG Desktop Portal file dialogs are supported
MRVIEWER_API bool isPortalFileDialogSupported();

/// Open XDG Desktop Portal file dialog
MRVIEWER_API std::vector<std::filesystem::path> runPortalFileDialog( const MR::FileDialog::Parameters& params );

} // namespace MR::detail
#endif
