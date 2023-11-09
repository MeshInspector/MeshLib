#pragma once
#include "MRRenderGLHelpers.h"

namespace MR
{
// Class for rendering 3d scene into texture
class SceneTextureGL
{
public:
    // clears this framebuffer and binds it as main rendering target
    void bind( bool clear );
    // binds default framebuffer (and read/draw framebuffers)
    void unbind();
    // if present-> del();gen();
    // otherwise just gen()
    // msaaPow - 2^msaaPow samples, msaaPow < 0 - use same default amaunt of samples
    void reset( const Vector2i& size, int msaaPow );
    // copy texture so draw() can render it
    void copyTexture();
    // renders texture
    void draw();
private:
    FramebufferData fd_;
    QuadTextureVertexObject qt_;
};

}