#include "MRSceneTextureGL.h"
#include "MRGladGlfw.h"
#include "MRGLMacro.h"
#include "MRGLStaticHolder.h"
#include "MRViewer.h"

namespace MR
{

void SceneTextureGL::bind( bool clear )
{
    fd_.bind( clear );
}

void SceneTextureGL::unbind()
{
    fd_.bindDefault();
}

void SceneTextureGL::reset( const Vector2i& size, int msaa )
{
    if ( fd_.getTexture() != GlTexture2::NO_TEX )
    {
        fd_.del();
        qt_.del();
    }
    fd_.gen( size, msaa );
    qt_.gen();
}

void SceneTextureGL::copyTexture()
{
    fd_.copyTextureBindDef();
}

void SceneTextureGL::draw()
{
#ifndef __EMSCRIPTEN__
    GL_EXEC( glDisable( GL_MULTISAMPLE ) );
#endif

    const auto& size = fd_.getSize();
    GL_EXEC( glViewport( 0, 0, size.x, size.y ) );
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::SimpleOverlayQuad );
    GL_EXEC( glUseProgram( shader ) );

    qt_.bind();
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    setTextureWrapType( WrapType::Mirror );
    setTextureFilterType( FilterType::Discrete );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, fd_.getTexture() ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "depth" ), 0.5f ) );
    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "viewportSize" ), float( size.x ), float( size.y ) ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "pixels" ), 0 ) );
    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, 6 ) );

#ifndef __EMSCRIPTEN__
    GL_EXEC( glEnable( GL_MULTISAMPLE ) );
#endif
}

}