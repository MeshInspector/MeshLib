#pragma once

#include "MRViewerFwd.h"

namespace MR
{

/// Fill the current OpenGL framebuffer with given color
MRVIEWER_API void fillFramebuffer( const Color& color );

/// Render to an image using OpenGL texture
MRVIEWER_API Image renderToImage(
    const Vector2i& resolution,
    const std::function<void ( void )>& renderFunc
);

} // namespace MR
