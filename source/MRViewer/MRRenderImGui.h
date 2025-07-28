#pragma once

#include "MRViewerFwd.h"

namespace MR
{

/// Render ImGui to the current OpenGL framebuffer. Uses a new ImGui context with shared fonts and style, thus can be called during the UI rendering.
/// \param resolution - ImGui's display size
/// \param configureFunc - optional callback to configure the ImGui context, called before ImGui::NewFrame()
/// \param drawFunc - callback to execute ImGui draw functions, called immediately after ImGui::NewFrame() and before ImGui::Render()
MRVIEWER_API void renderImGui(
    const Vector2i& resolution,
    const std::function<void ()>& configureFunc,
    const std::function<void ()>& drawFunc
);

} // namespace MR
