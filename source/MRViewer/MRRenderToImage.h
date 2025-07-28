#pragma once

#include "MRViewerFwd.h"

#include <optional>

namespace MR
{

/// Render to an image using OpenGL texture
/// \param resolution - image resolution
/// \param backgroundColor - image background color, can be omitted if your draw function fills the background by itself
/// \param drawFunc - callback for OpenGL drawing functions
/// \code
/// auto image = renderToImage( imageSize, {}, []
/// {
///     drawScene();
///     drawUI();
/// } );
/// \endcode
MRVIEWER_API Image renderToImage(
    const Vector2i& resolution,
    const std::optional<Color>& backgroundColor,
    const std::function<void ()>& drawFunc
);

} // namespace MR
