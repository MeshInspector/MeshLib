#pragma once

#include "exports.h"

#include <MRMesh/MRExpected.h>

namespace MR
{

// returns data in clipboard
[[nodiscard]] MRVIEWER_API Expected<std::string> GetClipboardText();

// sets data in clipboard
MRVIEWER_API Expected<void> SetClipboardText( const std::string& text );

} // namespace MR
