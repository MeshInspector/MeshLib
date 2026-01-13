#include "MRRenderImGui.h"

#include "MRGladGlfw.h"
#include "MRImGui.h"
#include "MRViewer.h"

#include "backends/imgui_impl_opengl3.h"
#include <imgui_internal.h>

namespace MR
{

void renderImGui( const Vector2i& resolution, const std::function<void()>& configureFunc,
    const std::function<void()>& drawFunc )
{
    auto& viewer = Viewer::instanceRef();
    if ( !viewer.isGLInitialized() )
        return;

    // backup ImGui context
    auto* backupCtx = ImGui::GetCurrentContext();
    auto* ctx = ImGui::CreateContext( backupCtx->IO.Fonts );
    ctx->IO.BackendFlags = ImGui::GetIO().BackendFlags;
    ctx->IO.BackendLanguageUserData = ImGui::GetIO().BackendLanguageUserData;
    ctx->IO.BackendPlatformName = ImGui::GetIO().BackendPlatformName;
    ctx->IO.BackendPlatformUserData = ImGui::GetIO().BackendPlatformUserData;
    ctx->IO.BackendRendererName = ImGui::GetIO().BackendRendererName;
    ctx->IO.BackendRendererUserData = ImGui::GetIO().BackendRendererUserData;
    ctx->Style = backupCtx->Style;
    ImGui::SetCurrentContext( ctx );

    // configure ImGui context
    ImGui::GetIO().DisplaySize = { (float)resolution.x, (float)resolution.y };
    if ( viewer.hasScaledFramebuffer() )
        ImGui::GetIO().DisplayFramebufferScale = { 1.f, 1.f };
    ImGui::GetIO().IniFilename = nullptr;
    if ( configureFunc )
        configureFunc();

    // render ImGui
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();
    drawFunc();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );

    // restore ImGui context
    ctx->IO.BackendFlags = 0;
    ctx->IO.BackendLanguageUserData = nullptr;
    ctx->IO.BackendPlatformUserData = nullptr;
    ctx->IO.BackendRendererUserData = nullptr;
    ImGui::SetCurrentContext( backupCtx );
    ImGui::DestroyContext( ctx );
}

} // namespace MR
