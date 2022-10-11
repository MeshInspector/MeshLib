#include "MRShadowsGL.h"
#include "MRViewer.h"
#include "MRGLMacro.h"
#include "MRGladGlfw.h"
#include "MRGLStaticHolder.h"

namespace MR
{

ShadowsGL::~ShadowsGL()
{
    if ( enabled_ )
        enable( false );
    if ( preDrawConnection_.connected() )
        preDrawConnection_.disconnect();
    if ( postDrawConnection_.connected() )
        postDrawConnection_.disconnect();
}

void ShadowsGL::enable( bool on )
{
    if ( !getViewerInstance().isGLInitialized() )
        return;
    if ( on == enabled_ )
        return;
    enabled_ = on;
    if ( on )
    {
        preDrawConnection_ = getViewerInstance().preDrawPostViewportSignal.connect( MAKE_SLOT( &ShadowsGL::preDraw_ ), boost::signals2::at_back );
        postDrawConnection_ = getViewerInstance().postDrawPreViewportSignal.connect( MAKE_SLOT( &ShadowsGL::postDraw_ ), boost::signals2::at_front );
    }
    else
    {
        preDrawConnection_.disconnect();
        postDrawConnection_.disconnect();
    }
}

void ShadowsGL::preDraw_()
{
    int width, height;
    glfwGetFramebufferSize( getViewerInstance().window, &width, &height );

    int samples;
    GL_EXEC( glGetIntegerv( GL_SAMPLES, &samples ) );

    // Create an initial multisampled framebuffer
    GL_EXEC( glGenFramebuffers( 1, &framebufferId_ ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, framebufferId_ ) );
    // create a multisampled color attachment texture
    GL_EXEC( glGenTextures( 1, &textureColorBufferMultiSampled_ ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D_MULTISAMPLE, textureColorBufferMultiSampled_ ) );
    GL_EXEC( glTexImage2DMultisample( GL_TEXTURE_2D_MULTISAMPLE, samples, GL_RGBA, width, height, GL_TRUE ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D_MULTISAMPLE, 0 ) );
    GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, textureColorBufferMultiSampled_, 0 ) );
    // create a (also multisampled) renderbuffer object for depth and stencil attachments
    GL_EXEC( glGenRenderbuffers( 1, &renderBufferObj_ ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, renderBufferObj_ ) );
    GL_EXEC( glRenderbufferStorageMultisample( GL_RENDERBUFFER, samples, GL_DEPTH_COMPONENT32F, width, height ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, 0 ) );
    GL_EXEC( glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderBufferObj_ ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );

    // configure second post-processing framebuffer
    GL_EXEC( glGenFramebuffers( 1, &intermediateFBO_ ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, intermediateFBO_ ) );
    // create a color attachment texture
    GL_EXEC( glGenTextures( 1, &screenColorTexture_ ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, screenColorTexture_ ) );
    GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR ) );
    GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, screenColorTexture_, 0 ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );

    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, intermediateFBO_ ) );
    // create a color attachment texture
    GL_EXEC( glGenTextures( 1, &screenDepthTexture_ ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, screenDepthTexture_ ) );
    GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR ) );
    GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, screenDepthTexture_, 0 ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );


    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, framebufferId_ ) );

    // Clear the buffer
    unsigned int cClearValue[4] = { 0x0,0x0,0x0,0x0 };
    GL_EXEC( glClearBufferuiv( GL_COLOR, 0, cClearValue ) );
    GL_EXEC( glClear( GL_DEPTH_BUFFER_BIT ) );
}

void ShadowsGL::postDraw_()
{
    int width, height;
    glfwGetFramebufferSize( getViewerInstance().window, &width, &height );
    GL_EXEC( glBindFramebuffer( GL_READ_FRAMEBUFFER, framebufferId_ ) );
    GL_EXEC( glBindFramebuffer( GL_DRAW_FRAMEBUFFER, intermediateFBO_ ) );
    GL_EXEC( glBlitFramebuffer( 0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST ) );

    GL_EXEC( glBindFramebuffer( GL_DRAW_FRAMEBUFFER, 0 ) );
    GL_EXEC( glBindFramebuffer( GL_READ_FRAMEBUFFER, 0 ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );
    
    // Draw shadows and main texture
    constexpr GLfloat quad[18] =
    {
        -1.0f, -1.0f, 0.99f,
        1.0f, -1.0f, 0.99f,
        -1.0f,  1.0f, 0.99f,
        -1.0f,  1.0f, 0.99f,
        1.0f, -1.0f, 0.99f,
        1.0f,  1.0f, 0.99f
    };
#ifndef __EMSCRIPTEN__
    GL_EXEC( glDisable( GL_MULTISAMPLE ) );
#endif
    GL_EXEC( glViewport( 0, 0, width, height ) );

    GLuint quadVao = 0;
    GLuint quadVbo = 0;
    GL_EXEC( glGenVertexArrays( 1, &quadVao ) );
    GL_EXEC( glGenBuffers( 1, &quadVbo ) );

    // draw shadows
    GL_EXEC( glBindVertexArray( quadVao ) );

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::ShadowOverlayQuad );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, quadVbo ) );
    GL_EXEC( glBufferData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * 18, quad, GL_STATIC_DRAW ) );

    GL_EXEC( glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( 0 ) );

    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "color" ), shadowColor.x, shadowColor.y, shadowColor.z, shadowColor.w ) );
    GL_EXEC( glUniform2i( glGetUniformLocation( shader, "shift" ), shadowShift.x, shadowShift.y ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "blurRadius" ), blurRadius ) );

    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, screenColorTexture_ ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "pixels" ), 0 ) );

    GL_EXEC( glBindVertexArray( quadVao ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, 6 ) );

    // draw main texture
    GL_EXEC( glBindVertexArray( quadVao ) );

    shader = GLStaticHolder::getShaderId( GLStaticHolder::SimpleOverlayQuad );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, quadVbo ) );
    GL_EXEC( glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( 0 ) );

    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, screenColorTexture_ ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "pixels" ), 0 ) );


    GL_EXEC( glActiveTexture( GL_TEXTURE1 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, screenDepthTexture_ ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "depth" ), 1 ) );

    GL_EXEC( glBindVertexArray( quadVao ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, 6 ) );
#ifndef __EMSCRIPTEN__
    GL_EXEC( glEnable( GL_MULTISAMPLE ) );
#endif

    GL_EXEC( glDeleteVertexArrays( 1, &quadVao ) );
    GL_EXEC( glDeleteBuffers( 1, &quadVbo ) );

    // Clean up
    GL_EXEC( glDeleteTextures( 1, &screenDepthTexture_ ) );
    GL_EXEC( glDeleteTextures( 1, &screenColorTexture_ ) );
    GL_EXEC( glDeleteTextures( 1, &textureColorBufferMultiSampled_ ) );
    GL_EXEC( glDeleteFramebuffers( 1, &framebufferId_ ) );
    GL_EXEC( glDeleteFramebuffers( 1, &intermediateFBO_ ) );
    GL_EXEC( glDeleteRenderbuffers( 1, &renderBufferObj_ ) );
}

}