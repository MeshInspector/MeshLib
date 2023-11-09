#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRVector4.h"
#include "MRMesh/MRColor.h"
#include "MRRenderGLHelpers.h"
#include <boost/signals2/connection.hpp>

namespace MR
{

// This class allows do draw shadows for objects in scene
// draws scene into texture buffer, than make shadow from:
// draw shadow and than draw texture with scene
class MRVIEWER_CLASS ShadowsGL
{
public:
    MR_ADD_CTOR_DELETE_MOVE( ShadowsGL );
    MRVIEWER_API ~ShadowsGL();

    // subscribe to viewer events on enable, unsubscribe on disable
    MRVIEWER_API void enable( bool on );
    bool isEnabled() const { return enabled_; }

    // shift in screen space
    const Vector2f& getShadowShift() const { return shadowShift_; }
    MRVIEWER_API void setShadowShift( const Vector2f& shift );

    const Vector4f& getShadowColor() const { return shadowColor_; }
    MRVIEWER_API void setShadowColor( const Vector4f& color );
    
    float getBlurRadius() const { return blurRadius_; }
    MRVIEWER_API void setBlurRadius( float radius );

    // value that describes blur quality, blur texture downscaling coefficient
    // (0,1] 1 - is maximum quality, but it can affect performance on embedded systems
    // 0.25 - recommended value
    float getQuality() const { return quality_; }
    MRVIEWER_API void setQuality( float quality );
private:
    // shift in screen space
    Vector2f shadowShift_ = Vector2f( 0.0, 0.0 );
    Vector4f shadowColor_ = Vector4f( Color::yellow() );
    float blurRadius_{ 40.0f };

    float quality_{ 0.25f };
    void preDraw_();
    void postDraw_();
    void postResize_( int x, int y );
    
    void drawLowSize_();
    void convolveX_();
    void convolveY_();
    void drawShadow_( bool convX );
    void drawTexture_( bool scene, bool downsample );

    boost::signals2::connection preDrawConnection_;
    boost::signals2::connection postDrawConnection_;
    boost::signals2::connection postResizeConnection_;

    Vector2i sceneSize_;
    Vector2i lowSize_;

    QuadTextureVertexObject quadObject_;

    FramebufferData sceneFramebuffer_;
    FramebufferData lowSizeFramebuffer_;
    FramebufferData convolutionXFramebuffer_;

    bool enabled_{ false };
};

}