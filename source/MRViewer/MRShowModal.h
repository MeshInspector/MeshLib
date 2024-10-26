#pragma once

#include "MRViewerFwd.h"
#include "MRNotificationType.h"

namespace MR
{

/// Check if menu is available and if it is, shows modal window
MRVIEWER_API void showModal( const std::string& error, NotificationType type );
inline void showError( const std::string& error )
{
    showModal( error, NotificationType::Error );
}

} //namespace MR
