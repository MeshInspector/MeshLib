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

    getViewerInstance().setSceneDirty();
    enabled_ = on;
    if ( on )
    {
        preDrawConnection_ = getViewerInstance().preDrawSignal.connect( MAKE_SLOT( &ShadowsGL::preDraw_ ), boost::signals2::at_back );
        postDrawConnection_ = getViewerInstance().postDrawPreViewportSignal.connect( MAKE_SLOT( &ShadowsGL::postDraw_ ), boost::signals2::at_front );
        postResizeConnection_ = getViewerInstance().postResizeSignal.connect( MAKE_SLOT( &ShadowsGL::postResize_ ), boost::signals2::at_back );
        
        glfwGetFramebufferSize( getViewerInstance().window, &sceneSize_.x, &sceneSize_.y );
        lowSize_ = Vector2i( Vector2f( sceneSize_ ) * quality_ );
        quadObject_.gen();
        sceneFramebuffer_.gen( sceneSize_, -1 );
        lowSizeFramebuffer_.gen( lowSize_, 0 );
        convolutionXFramebuffer_.gen( lowSize_, 0 );
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

void ShadowsGL::setShadowShift( const Vector2f& shift )
{
    if ( shadowShift_ == shift )
        return;
    shadowShift_ = shift;
    getViewerInstance().setSceneDirty();
}

void ShadowsGL::setShadowColor( const Vector4f& color )
{
    if ( shadowColor_ == color )
        return;
    shadowColor_ = color;
    getViewerInstance().setSceneDirty();
}

void ShadowsGL::setBlurRadius( float radius )
{
    if ( blurRadius_ == radius )
        return;
    blurRadius_ = radius;
    getViewerInstance().setSceneDirty();
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
        
    sceneFramebuffer_.gen( sceneSize_, -1 );
    lowSizeFramebuffer_.gen( lowSize_, 0 );
    convolutionXFramebuffer_.gen( lowSize_, 0 );
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
    sceneFramebuffer_.copyTextureBindDef();
    drawLowSize_(); // draw scene texture in low size for further convolution
    convolveX_(); // draw shadow with x convolution to other texture (low res)
    convolveY_(); // draw shadow with y convolution to other texture (low res)
    getViewerInstance().bindSceneTexture( true ); // bind default scene texture
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
    lowSizeFramebuffer_.copyTextureBindDef();
}

void ShadowsGL::convolveX_()
{
    convolutionXFramebuffer_.bind();
    drawShadow_( true );
    convolutionXFramebuffer_.copyTextureBindDef();
}

void ShadowsGL::convolveY_()
{
    lowSizeFramebuffer_.bind(); // reuse this framebuffer for y conv
    drawShadow_( false );
    lowSizeFramebuffer_.copyTextureBindDef();
}

void ShadowsGL::drawShadow_( bool convX )
{
    GL_EXEC( glViewport( 0, 0, lowSize_.x, lowSize_.y ) );
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::ShadowOverlayQuad );
    GL_EXEC( glUseProgram( shader ) );

    quadObject_.bind();
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "color" ), shadowColor_.x, shadowColor_.y, shadowColor_.z, shadowColor_.w ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "blurRadius" ), blurRadius_ * quality_ ) );
    GL_EXEC( glUniform2f( glGetUniformLocation( shader, "shift" ), shadowShift_.x * quality_, shadowShift_.y * quality_) );
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
    
    lowSizeFramebuffer_.gen( lowSize_, 0 );
    convolutionXFramebuffer_.gen( lowSize_, 0 );

    getViewerInstance().setSceneDirty();
}

}