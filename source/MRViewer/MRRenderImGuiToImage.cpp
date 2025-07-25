#include "MRRenderImGuiToImage.h"

#include "MRImGui.h"
#include "MRCommandLoop.h"
#include "MRRenderGLHelpers.h"
#include "MRViewer.h"

#include "MRMesh/MRImage.h"

#include "backends/imgui_impl_opengl3.h"

namespace MR
{

void renderImGuiToImage( Vector2i resolution, Color backgroundColor, std::function<void ( void )> renderFunc, std::function<void ( Image )> imageCallback )
{
    CommandLoop::appendCommand( [=]
    {
        auto& viewer = Viewer::instanceRef();
        if ( !viewer.isGLInitialized() )
            return;

        FramebufferData fd;
        fd.gen( resolution, 1 );
        fd.bind( false );

        GL_EXEC( glClearColor( backgroundColor.r, backgroundColor.g, backgroundColor.b, backgroundColor.a ) );
        GL_EXEC( glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ) );

        // render ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui::GetIO().DisplaySize = { (float)resolution.x, (float)resolution.y };
        if ( viewer.hasScaledFramebuffer() )
            ImGui::GetIO().DisplayFramebufferScale = { 1.f, 1.f };
        ImGui::NewFrame();
        renderFunc();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );

        fd.copyTextureBindDef();
        fd.bindTexture();

        // copy texture to image
        Image result;
        result.resolution = resolution;
        result.pixels.resize( (size_t)resolution.x * resolution.y );
        GL_EXEC( glGetTexImage( GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)result.pixels.data() ) );

        fd.del();

        viewer.bindSceneTexture( true );

        imageCallback( std::move( result ) );
    } );
}

} // namespace MR
