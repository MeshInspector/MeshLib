#pragma once

#include "MRViewerFwd.h"

namespace MR
{

/// Render ImGui to an image using OpenGL texture. Called between frames' rendering in a separate context.
/// \param resolution - image resolution, also sets the ImGui's display size
/// \param backgroundColor - image background color
/// \param renderFunc - callback to execute ImGui draw functions, called immediately after ImGui::NewFrame() and before ImGui::Render()
/// \param imageCallback - callback with the output image
MRVIEWER_API void renderImGuiToImage(
    Vector2i resolution,
    Color backgroundColor,
    std::function<void ( void )> renderFunc,
    std::function<void ( Image )> imageCallback
);

} // namespace MR
