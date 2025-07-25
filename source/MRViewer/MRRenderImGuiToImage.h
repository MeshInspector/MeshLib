#pragma once

#include "MRViewerFwd.h"

namespace MR
{

/// Render ImGui to an image using OpenGL texture. Uses a new ImGui context with the shared font atlas.
/// \param resolution - image resolution, also sets the ImGui's display size
/// \param backgroundColor - image background color
/// \param renderFunc - callback to execute ImGui draw functions, called immediately after ImGui::NewFrame() and before ImGui::Render()
MRVIEWER_API Image renderImGuiToImage(
    const Vector2i& resolution,
    const Color& backgroundColor,
    const std::function<void ( void )>& renderFunc
);

} // namespace MR
