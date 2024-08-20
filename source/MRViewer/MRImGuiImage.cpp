#include "MRImGuiImage.h"
#include "MRGLMacro.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"

namespace MR
{

ImGuiImage::ImGuiImage()
{
}

ImGuiImage::~ImGuiImage()
{
}

void ImGuiImage::update( const MeshTexture& texture )
{
    texture_ = texture;
    bind_();
}

void ImGuiImage::bind_()
{
    if ( !getViewerInstance().isGLInitialized() )
        return;
    glTex_.loadData(
        { 
            .resolution = GlTexture2::ToResolution( texture_.resolution ),
            .internalFormat = GL_RGBA, 
            .format = GL_RGBA, 
            .type = GL_UNSIGNED_BYTE, 
            .wrap = texture_.wrap, 
            .filter = texture_.filter
        },
        texture_.pixels );
}


}
