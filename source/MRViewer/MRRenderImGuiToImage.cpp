#include "MRRenderImGuiToImage.h"

#include "MRImGui.h"
#include "MRCommandLoop.h"
#include "MRRenderGLHelpers.h"
#include "MRViewer.h"

#include "MRMesh/MRImage.h"

#include "backends/imgui_impl_opengl3.h"

namespace MR
{

Image renderImGuiToImage( const Vector2i& resolution, const Color& backgroundColor, const std::function<void ( void )>& renderFunc )
{
    auto& viewer = Viewer::instanceRef();
    if ( !viewer.isGLInitialized() )
        return {};

    FramebufferData fd;
    fd.gen( resolution, 1 );
    fd.bind( false );

    GL_EXEC( glClearColor( backgroundColor.r, backgroundColor.g, backgroundColor.b, backgroundColor.a ) );
    GL_EXEC( glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ) );

    // backup ImGui context
    auto* backupCtx = ImGui::GetCurrentContext();
    auto* ctx = ImGui::CreateContext( backupCtx->IO.Fonts );
    ctx->Style = backupCtx->Style;
    ImGui::SetCurrentContext( ctx );

    // configure ImGui context
    ImGui::GetIO().IniFilename = nullptr;

    // render ImGui
    #ifdef __EMSCRIPTEN__
        const char* glsl_version = "#version 300 es";
    #else
        const char* glsl_version = "#version 150";
    #endif
    ImGui_ImplOpenGL3_Init( glsl_version );
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::GetIO().DisplaySize = { (float)resolution.x, (float)resolution.y };
    if ( viewer.hasScaledFramebuffer() )
        ImGui::GetIO().DisplayFramebufferScale = { 1.f, 1.f };
    ImGui::NewFrame();
    renderFunc();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );

    // restore ImGui context
    ImGui::SetCurrentContext( backupCtx );
    ImGui::DestroyContext( ctx );

    fd.copyTextureBindDef();
    fd.bindTexture();

    // copy texture to image
    Image result;
    result.resolution = resolution;
    result.pixels.resize( (size_t)resolution.x * resolution.y );
#ifdef __EMSCRIPTEN__
    GLuint fbo;
    GL_EXEC( glGenFramebuffers( 1, &fbo ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, fbo ) );
    GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fd.getTexture(), 0 ) );

    GL_EXEC( glReadPixels( 0, 0, newRes.x, newRes.y, GL_RGBA, GL_UNSIGNED_BYTE, (void*)result.pixels.data() ) );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
    glDeleteFramebuffers( 1, &fbo );
#else
    GL_EXEC( glGetTexImage( GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)result.pixels.data() ) );
#endif

    fd.del();

    viewer.bindSceneTexture( true );

    return result;
}

} // namespace MR
