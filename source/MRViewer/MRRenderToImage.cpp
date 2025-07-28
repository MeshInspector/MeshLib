#include "MRRenderToImage.h"

#include "MRRenderGLHelpers.h"
#include "MRViewer.h"

#include "MRMesh/MRImage.h"

namespace
{

using namespace MR;

void fillFramebuffer( const Color& color )
{
    GL_EXEC( glClearColor( color.r, color.g, color.b, color.a ) );
    GL_EXEC( glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ) );
}

} // namespace

namespace MR
{

Image renderToImage( const Vector2i& resolution, const std::optional<Color>& backgroundColor, const std::function<void()>& drawFunc )
{
    auto& viewer = Viewer::instanceRef();
    if ( !viewer.isGLInitialized() )
        return {};

    FramebufferData fd;
    fd.gen( resolution, getMSAAPow( viewer.getRequestedMSAA() ) );
    fd.bind( false );

    if ( backgroundColor )
        fillFramebuffer( *backgroundColor );

    drawFunc();

    fd.copyTextureBindDef();
    fd.bindTexture();

    Image result;
    result.resolution = resolution;
    result.pixels.resize( (size_t)resolution.x * resolution.y );

    // copy texture to image
#ifdef __EMSCRIPTEN__
    GLuint fbo;
    GL_EXEC( glGenFramebuffers( 1, &fbo ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, fbo ) );
    GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fd.getTexture(), 0 ) );

    GL_EXEC( glReadPixels( 0, 0, resolution.x, resolution.y, GL_RGBA, GL_UNSIGNED_BYTE, (void*)result.pixels.data() ) );

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
