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
        preDrawConnection_ = getViewerInstance().preDrawSignal.connect( MAKE_SLOT( &ShadowsGL::preDraw_ ), boost::signals2::at_back );
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
    glfwGetFramebufferSize( getViewerInstance().window, &sceneSize_.x, &sceneSize_.y );
    if ( sceneSize_.x == 0 || sceneSize_.y == 0 )
        return;
    shadowSize_ = Vector2i( Vector2d( sceneSize_ ) / double( 2 * std::abs( blurRadius ) + 1 ) );

    int samples;
    GL_EXEC( glGetIntegerv( GL_SAMPLES, &samples ) );

    // Create an initial multisampled framebuffer
    GL_EXEC( glGenFramebuffers( 1, &sceneFramebuffer_ ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, sceneFramebuffer_ ) );

    // create a multisampled color renderbuffer
    GL_EXEC( glGenRenderbuffers( 1, &sceneColorRenderbufferMultisampled_ ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, sceneColorRenderbufferMultisampled_ ) );
    GL_EXEC( glRenderbufferStorageMultisample( GL_RENDERBUFFER, samples, GL_RGBA8, sceneSize_.x, sceneSize_.y ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, 0 ) );
    GL_EXEC( glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, sceneColorRenderbufferMultisampled_ ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );

    // create a (also multisampled) renderbuffer object for depth attachments
    GL_EXEC( glGenRenderbuffers( 1, &sceneDepthRenderbufferMultisampled_ ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, sceneDepthRenderbufferMultisampled_ ) );
    GL_EXEC( glRenderbufferStorageMultisample( GL_RENDERBUFFER, samples, GL_DEPTH_COMPONENT24, sceneSize_.x, sceneSize_.y ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, 0 ) );
    GL_EXEC( glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, sceneDepthRenderbufferMultisampled_ ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );

    // configure second post-processing framebuffer
    GL_EXEC( glGenFramebuffers( 1, &sceneCopyFramebuffer_ ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, sceneCopyFramebuffer_ ) );
    // create a color attachment texture
    GL_EXEC( glGenTextures( 1, &sceneResTexture_ ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, sceneResTexture_ ) );
    GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, sceneSize_.x, sceneSize_.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR ) );
    GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sceneResTexture_, 0 ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );

    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, sceneFramebuffer_ ) );

    // Clear the buffer
    float cClearValue[4] = { 0.0f,0.0f,0.0f,0.0f };
    GL_EXEC( glClearBufferfv( GL_COLOR, 0, cClearValue ) );
    GL_EXEC( glClear( GL_DEPTH_BUFFER_BIT ) );
}

void ShadowsGL::postDraw_()
{
    if ( sceneSize_.x == 0 || sceneSize_.y == 0 )
        return;

    GL_EXEC( glBindFramebuffer( GL_READ_FRAMEBUFFER, sceneFramebuffer_ ) );
    GL_EXEC( glBindFramebuffer( GL_DRAW_FRAMEBUFFER, sceneCopyFramebuffer_ ) );
    GL_EXEC( glBlitFramebuffer( 0, 0, sceneSize_.x, sceneSize_.y, 0, 0, sceneSize_.x, sceneSize_.y, GL_COLOR_BUFFER_BIT, GL_NEAREST ) );

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
    GL_EXEC( glViewport( 0, 0, sceneSize_.x, sceneSize_.y ) );

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
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "blurRadius" ), blurRadius ) );
    GL_EXEC( glUniform2i( glGetUniformLocation( shader, "shift" ), shadowShift.x, shadowShift.y ) );
    
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, sceneResTexture_ ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "pixels" ), 0 ) );
    
    GL_EXEC( glBindVertexArray( quadVao ) );
    
    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, 6 ) );

    // draw main texture
    GL_EXEC( glBindVertexArray( quadVao ) );

    shader = GLStaticHolder::getShaderId( GLStaticHolder::SimpleOverlayQuad );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, quadVbo ) );
    GL_EXEC( glBufferData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * 18, quad, GL_STATIC_DRAW ) );

    GL_EXEC( glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( 0 ) );

    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, sceneResTexture_ ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "pixels" ), 0 ) );

    GL_EXEC( glBindVertexArray( quadVao ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, 6 ) );
#ifndef __EMSCRIPTEN__
    GL_EXEC( glEnable( GL_MULTISAMPLE ) );
#endif

    GL_EXEC( glDeleteVertexArrays( 1, &quadVao ) );
    GL_EXEC( glDeleteBuffers( 1, &quadVbo ) );

    // Clean up
    GL_EXEC( glDeleteTextures( 1, &sceneResTexture_ ) );
    GL_EXEC( glDeleteFramebuffers( 1, &sceneFramebuffer_ ) );
    GL_EXEC( glDeleteFramebuffers( 1, &sceneCopyFramebuffer_ ) );
    GL_EXEC( glDeleteRenderbuffers( 1, &sceneDepthRenderbufferMultisampled_ ) );
    GL_EXEC( glDeleteRenderbuffers( 1, &sceneColorRenderbufferMultisampled_ ) );
}

}