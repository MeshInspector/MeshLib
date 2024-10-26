#pragma once

#include "MRViewerFwd.h"
#include <MRMesh/MRViewportId.h>

// This is lightweight header, pleas include it instead of "MRViewer.h" whenever possible

namespace MR
{

/// returns global instance of Viewer class
[[nodiscard]] MRVIEWER_API Viewer& getViewerInstance();

/// Increment number of forced frames to redraw in event loop
/// if `swapOnLastOnly` only last forced frame will be present on screen and all previous will not
MRVIEWER_API void incrementForceRedrawFrames( int i = 1, bool swapOnLastOnly = false );

/// Return the current viewport, or the viewport corresponding to a given unique identifier
/// \param viewportId unique identifier corresponding to the desired viewport (current viewport if 0)
[[nodiscard]] MRVIEWER_API Viewport& viewport( ViewportId viewportId = {} );

} //namespace MR
