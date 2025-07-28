#pragma once

#include "MRViewerFwd.h"

namespace MR
{

/// Render to an image using OpenGL texture
MRVIEWER_API Image renderToImage(
    const Vector2i& resolution,
    const std::optional<Color>& backgroundColor,
    const std::function<void ( void )>& renderFunc
);

} // namespace MR
