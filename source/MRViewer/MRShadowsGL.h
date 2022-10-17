#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRVector4.h"
#include "MRMesh/MRColor.h"
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
    Vector2f shadowShift = Vector2f( 0.0, 0.0 );
    Vector4f shadowColor = Vector4f( Color::yellow() );
    float blurRadius{ 40.0f };
    // value that describes blur quality, blur texture downscaling coefficient
    // (0,1] 1 - is maximum quality, but it can affect performance on embedded systems
    // 0.25 - recomended value
    float getQuality() const { return quality_; }
    MRVIEWER_API void setQuality( float quality );
private:
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

    class FramebufferData
    {
    public:
        void gen( const Vector2i& size, bool multisample );
        void bind();
        void copyTexture();
        void del();
        unsigned getTexture() const { return resTexture_; }
    private:
        void resize_( const Vector2i& size, bool multisample );

        unsigned mainFramebuffer_{ 0 };
        unsigned colorRenderbuffer_{ 0 };
        unsigned depthRenderbuffer_{ 0 };
        unsigned copyFramebuffer_{ 0 };
        unsigned resTexture_{ 0 };
        Vector2i size_;
    };

    class QuadTextureVertexObject
    {
    public:
        void gen();
        void bind();
        void del();
    private:
        unsigned vao_;
        unsigned vbo_;
    } quadObject_;

    FramebufferData sceneFramebuffer_;
    FramebufferData lowSizeFramebuffer_;
    FramebufferData convolveXFramebuffer_;

    bool enabled_{ false };
};

}