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

    boost::signals2::connection preDrawConnection_;
    boost::signals2::connection postDrawConnection_;

    Vector2i sceneSize_;

    unsigned int sceneFramebuffer_{ 0 };
    unsigned int sceneColorRenderbufferMultisampled_{ 0 };
    unsigned int sceneDepthRenderbufferMultisampled_{ 0 };
    unsigned int sceneCopyFramebuffer_{ 0 };
    unsigned int sceneResTexture_{ 0 };

    bool enabled_{ false };
};

}