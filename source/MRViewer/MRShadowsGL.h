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
    float blurRadius{ 3.0f };
private:
    void preDraw_();
    void postDraw_();

    void convolveX_();
    void drawShadow_( bool convX );
    void drawScene_();

    boost::signals2::connection preDrawConnection_;
    boost::signals2::connection postDrawConnection_;

    Vector2i sceneSize_;

    class FramebufferData
    {
    public:
        void gen( const Vector2i& size, bool multisample );
        void copyTexture();
        void del();
        unsigned getTexture() const { return resTexture_; }
    private:
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
    FramebufferData convolveFramebuffer_;

    bool enabled_{ false };
};

}