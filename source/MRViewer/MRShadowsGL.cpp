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

    sceneFramebuffer_.gen( sceneSize_, true );
}

void ShadowsGL::postDraw_()
{
    if ( sceneSize_.x == 0 || sceneSize_.y == 0 )
        return;

    quadObject_.gen();
    convolveX_(); // draw shadow with x convolution to other texture

#ifndef __EMSCRIPTEN__
    GL_EXEC( glDisable( GL_MULTISAMPLE ) );
#endif
    drawShadow_( false );
    drawScene_();
    
#ifndef __EMSCRIPTEN__
    GL_EXEC( glEnable( GL_MULTISAMPLE ) );
#endif

    quadObject_.del();
    sceneFramebuffer_.del();
    convolveFramebuffer_.del();
}


void ShadowsGL::convolveX_()
{
    sceneFramebuffer_.copyTexture();
    convolveFramebuffer_.gen( sceneSize_, false );
    drawShadow_( true );
    convolveFramebuffer_.copyTexture();
}

void ShadowsGL::drawShadow_( bool convX )
{
    GL_EXEC( glViewport( 0, 0, sceneSize_.x, sceneSize_.y ) );
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::ShadowOverlayQuad );
    GL_EXEC( glUseProgram( shader ) );

    quadObject_.bind();
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "color" ), shadowColor.x, shadowColor.y, shadowColor.z, shadowColor.w ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "blurRadius" ), blurRadius ) );
    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "shift" ), shadowShift.x, shadowShift.y ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "convX" ), convX ) );

    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    if ( convX )
    {
        // draw scene texture to texture with X convolution
        GL_EXEC( glBindTexture( GL_TEXTURE_2D, sceneFramebuffer_.getTexture() ) );
    }
    else
    {
        // draw X convolution texture to screen
        GL_EXEC( glBindTexture( GL_TEXTURE_2D, convolveFramebuffer_.getTexture() ) );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "pixels" ), 0 ) );
    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, 6 ) );
}

void ShadowsGL::drawScene_()
{
    GL_EXEC( glViewport( 0, 0, sceneSize_.x, sceneSize_.y ) );
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::SimpleOverlayQuad );
    GL_EXEC( glUseProgram( shader ) );

    quadObject_.bind();
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, sceneFramebuffer_.getTexture() ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "pixels" ), 0 ) );
    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, 6 ) );
}

void ShadowsGL::FramebufferData::gen( const Vector2i& size, bool multisample )
{
    size_ = size;
    int samples = 0;
    if ( multisample )
    {
        GL_EXEC( glGetIntegerv( GL_SAMPLES, &samples ) );
    }
    // Create an initial multisampled framebuffer
    GL_EXEC( glGenFramebuffers( 1, &mainFramebuffer_ ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, mainFramebuffer_ ) );

    // create a color renderbuffer
    GL_EXEC( glGenRenderbuffers( 1, &colorRenderbuffer_ ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, colorRenderbuffer_ ) );
    if ( multisample )
    {
        GL_EXEC( glRenderbufferStorageMultisample( GL_RENDERBUFFER, samples, GL_RGBA8, size.x, size.y ) );
    }
    else
    {
        GL_EXEC( glRenderbufferStorage( GL_RENDERBUFFER, GL_RGBA8, size.x, size.y ) );
    }
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, 0 ) );
    GL_EXEC( glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorRenderbuffer_ ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );

    // create a renderbuffer object for depth attachments
    GL_EXEC( glGenRenderbuffers( 1, &depthRenderbuffer_ ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, depthRenderbuffer_ ) );
    if ( multisample )
    {
        GL_EXEC( glRenderbufferStorageMultisample( GL_RENDERBUFFER, samples, GL_DEPTH_COMPONENT24, size.x, size.y ) );
    }
    else
    {
        GL_EXEC( glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, size.x, size.y ) );
    }
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, 0 ) );
    GL_EXEC( glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer_ ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );

    // configure second post-processing framebuffer
    GL_EXEC( glGenFramebuffers( 1, &copyFramebuffer_ ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, copyFramebuffer_ ) );
    // create a color attachment texture
    GL_EXEC( glGenTextures( 1, &resTexture_ ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, resTexture_ ) );
    GL_EXEC( glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR ) );
    GL_EXEC( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR ) );
    GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, resTexture_, 0 ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );

    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, mainFramebuffer_ ) );

    // Clear the buffer
    float cClearValue[4] = { 0.0f,0.0f,0.0f,0.0f };
    GL_EXEC( glClearBufferfv( GL_COLOR, 0, cClearValue ) );
    GL_EXEC( glClear( GL_DEPTH_BUFFER_BIT ) );
}

void ShadowsGL::FramebufferData::copyTexture()
{
    GL_EXEC( glBindFramebuffer( GL_READ_FRAMEBUFFER, mainFramebuffer_ ) );
    GL_EXEC( glBindFramebuffer( GL_DRAW_FRAMEBUFFER, copyFramebuffer_ ) );
    GL_EXEC( glBlitFramebuffer( 0, 0, size_.x, size_.y, 0, 0, size_.x, size_.y, GL_COLOR_BUFFER_BIT, GL_NEAREST ) );

    GL_EXEC( glBindFramebuffer( GL_DRAW_FRAMEBUFFER, 0 ) );
    GL_EXEC( glBindFramebuffer( GL_READ_FRAMEBUFFER, 0 ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );
}

void ShadowsGL::FramebufferData::del()
{
    GL_EXEC( glDeleteTextures( 1, &resTexture_ ) );
    GL_EXEC( glDeleteFramebuffers( 1, &mainFramebuffer_ ) );
    GL_EXEC( glDeleteFramebuffers( 1, &copyFramebuffer_ ) );
    GL_EXEC( glDeleteRenderbuffers( 1, &depthRenderbuffer_ ) );
    GL_EXEC( glDeleteRenderbuffers( 1, &colorRenderbuffer_ ) );
}

void ShadowsGL::QuadTextureVertexObject::gen()
{
    constexpr GLfloat quad[18] =
    {
        -1.0f, -1.0f, 0.99f,
        1.0f, -1.0f, 0.99f,
        -1.0f,  1.0f, 0.99f,
        -1.0f,  1.0f, 0.99f,
        1.0f, -1.0f, 0.99f,
        1.0f,  1.0f, 0.99f
    };
    GL_EXEC( glGenVertexArrays( 1, &vao_ ) );
    GL_EXEC( glGenBuffers( 1, &vbo_ ) );

    // draw shadows
    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, vbo_ ) );
    GL_EXEC( glBufferData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * 18, quad, GL_STATIC_DRAW ) );
}

void ShadowsGL::QuadTextureVertexObject::bind()
{
    GL_EXEC( glBindVertexArray( vao_ ) );
    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, vbo_ ) );
    GL_EXEC( glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( 0 ) );
}

void ShadowsGL::QuadTextureVertexObject::del()
{
    GL_EXEC( glDeleteVertexArrays( 1, &vao_ ) );
    GL_EXEC( glDeleteBuffers( 1, &vbo_ ) );
}

}