#pragma once

#include "MRViewerFwd.h"

#include <optional>

namespace MR
{

/// Render to an image using OpenGL texture
/// \param resolution - image resolution
/// \param backgroundColor - image background color, can be omitted if your draw function fills the background by itself
/// \param drawFunc - callback for OpenGL drawing functions
MRVIEWER_API Image renderToImage(
    const Vector2i& resolution,
    const std::optional<Color>& backgroundColor,
    const std::function<void ( void )>& drawFunc
);

} // namespace MR
