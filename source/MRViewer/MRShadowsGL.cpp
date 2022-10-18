#include "MRShadowsGL.h"
#include "MRViewer.h"
#include "MRGLMacro.h"
#include "MRGladGlfw.h"
#include "MRGLStaticHolder.h"
#include "MRCommandLoop.h"

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
        postResizeConnection_ = getViewerInstance().postResizeSignal.connect( MAKE_SLOT( &ShadowsGL::postResize_ ), boost::signals2::at_back );
        
        glfwGetFramebufferSize( getViewerInstance().window, &sceneSize_.x, &sceneSize_.y );
        lowSize_ = Vector2i( Vector2f( sceneSize_ ) * quality_ );
        quadObject_.gen();
        sceneFramebuffer_.gen( sceneSize_, true );
        lowSizeFramebuffer_.gen( lowSize_, false );
        convolutionXFramebuffer_.gen( lowSize_, false );
    }
    else
    {
        preDrawConnection_.disconnect();
        postDrawConnection_.disconnect();
        postResizeConnection_.disconnect();
        
        quadObject_.del();
        sceneFramebuffer_.del();
        convolutionXFramebuffer_.del();
        lowSizeFramebuffer_.del();
    }
}

void ShadowsGL::postResize_( int, int )
{
    glfwGetFramebufferSize( getViewerInstance().window, &sceneSize_.x, &sceneSize_.y );
    if ( sceneSize_.x == 0 || sceneSize_.y == 0 )
        return;
    lowSize_ = Vector2i( Vector2f( sceneSize_ ) * quality_ );
    sceneFramebuffer_.del();
    convolutionXFramebuffer_.del();
    lowSizeFramebuffer_.del();
        
    sceneFramebuffer_.gen( sceneSize_, true );
    lowSizeFramebuffer_.gen( lowSize_, false );
    convolutionXFramebuffer_.gen( lowSize_, false );
}

void ShadowsGL::preDraw_()
{
    sceneFramebuffer_.bind();
}

void ShadowsGL::postDraw_()
{
    if ( sceneSize_.x == 0 || sceneSize_.y == 0 )
        return;

#ifndef __EMSCRIPTEN__
    GL_EXEC( glDisable( GL_MULTISAMPLE ) );
#endif
    sceneFramebuffer_.copyTexture();
    drawLowSize_(); // draw scene texture in low size for further convolution
    convolveX_(); // draw shadow with x convolution to other texture (low res)
    convolveY_(); // draw shadow with y convolution to other texture (low res)
    drawTexture_( false, false ); // draw shadow in real size to main framebuffer
    drawTexture_( true, false ); // draw scene in real size to main framebuffer
    
#ifndef __EMSCRIPTEN__
    GL_EXEC( glEnable( GL_MULTISAMPLE ) );
#endif
}

void ShadowsGL::drawLowSize_()
{
    lowSizeFramebuffer_.bind();
    drawTexture_( true, true );
    lowSizeFramebuffer_.copyTexture();
}

void ShadowsGL::convolveX_()
{
    convolutionXFramebuffer_.bind();
    drawShadow_( true );
    convolutionXFramebuffer_.copyTexture();
}

void ShadowsGL::convolveY_()
{
    lowSizeFramebuffer_.bind(); // reuse this framebuffer for y conv
    drawShadow_( false );
    lowSizeFramebuffer_.copyTexture();
}

void ShadowsGL::drawShadow_( bool convX )
{
    GL_EXEC( glViewport( 0, 0, lowSize_.x, lowSize_.y ) );
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::ShadowOverlayQuad );
    GL_EXEC( glUseProgram( shader ) );

    quadObject_.bind();
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "color" ), shadowColor.x, shadowColor.y, shadowColor.z, shadowColor.w ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "blurRadius" ), blurRadius * quality_ ) );
    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "shift" ), shadowShift.x * quality_, shadowShift.y * quality_) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "convX" ), convX ) );

    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    if ( convX )
    {
        // draw scene texture to texture with X convolution
        GL_EXEC( glBindTexture( GL_TEXTURE_2D, lowSizeFramebuffer_.getTexture() ) );
    }
    else
    {
        // draw X convolution texture to screen
        GL_EXEC( glBindTexture( GL_TEXTURE_2D, convolutionXFramebuffer_.getTexture() ) );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "pixels" ), 0 ) );
    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, 6 ) );
}

void ShadowsGL::drawTexture_( bool scene, bool downsample )
{
    const auto& size = downsample ? lowSize_ : sceneSize_;
    GL_EXEC( glViewport( 0, 0, size.x, size.y ) );
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::SimpleOverlayQuad );
    GL_EXEC( glUseProgram( shader ) );

    quadObject_.bind();
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    if ( scene )
    {
        GL_EXEC( glBindTexture( GL_TEXTURE_2D, sceneFramebuffer_.getTexture() ) );
    }
    else
    {
        GL_EXEC( glBindTexture( GL_TEXTURE_2D, lowSizeFramebuffer_.getTexture() ) );
    }
    if ( scene )
    {
        GL_EXEC( glUniform1f( glGetUniformLocation( shader, "depth" ), 0.5f ) );
    }
    else
    {
        GL_EXEC( glUniform1f( glGetUniformLocation( shader, "depth" ), 0.99f ) );
    }
    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "viewportSize" ), float( size.x ), float( size.y ) ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "pixels" ), 0 ) );
    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, 6 ) );
}

void ShadowsGL::setQuality( float quality )
{
    if ( quality_ == quality )
        return;
    
    quality_ = quality;
    if ( quality_ <= 0.0f )
        quality_ = 0.125f;
    else if ( quality_ > 1.0f )
        quality_ = 1.0f;
    if ( !enabled_ )
        return;
    if ( sceneSize_.x == 0 || sceneSize_.y == 0 )
        return;

    lowSize_ = Vector2i( Vector2f( sceneSize_ ) * quality_ );
    convolutionXFramebuffer_.del();
    lowSizeFramebuffer_.del();
    
    lowSizeFramebuffer_.gen( lowSize_, false );
    convolutionXFramebuffer_.gen( lowSize_, false );
}

void ShadowsGL::FramebufferData::gen( const Vector2i& size, bool multisample )
{
    // Create an initial multisampled framebuffer
    GL_EXEC( glGenFramebuffers( 1, &mainFramebuffer_ ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, mainFramebuffer_ ) );

    // create a color renderbuffer
    GL_EXEC( glGenRenderbuffers( 1, &colorRenderbuffer_ ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, colorRenderbuffer_ ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, 0 ) );

    // create a renderbuffer object for depth attachments
    GL_EXEC( glGenRenderbuffers( 1, &depthRenderbuffer_ ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, depthRenderbuffer_ ) );
    GL_EXEC( glBindRenderbuffer( GL_RENDERBUFFER, 0 ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );

    // configure second post-processing framebuffer
    GL_EXEC( glGenFramebuffers( 1, &copyFramebuffer_ ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, copyFramebuffer_ ) );
    // create a color attachment texture
    resTexture_.gen();
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );

    resize_( size, multisample );
}

void ShadowsGL::FramebufferData::resize_( const Vector2i& size, bool multisample )
{
    size_ = size;
    int samples = 0;
    if ( multisample )
    {
        GL_EXEC( glGetIntegerv( GL_SAMPLES, &samples ) );
    }

    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, mainFramebuffer_ ) );

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
    
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, copyFramebuffer_ ) );

    resTexture_.loadData( { .resolution = size, .wrap = WrapType::Clamp,.filter = FilterType::Linear }, ( const char* ) nullptr );
    GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, resTexture_.getId(), 0 ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );
}

void ShadowsGL::FramebufferData::bind()
{
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
    resTexture_.del();
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