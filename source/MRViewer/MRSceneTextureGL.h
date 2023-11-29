#pragma once
#include "MRRenderGLHelpers.h"

namespace MR
{
// Class for rendering 3d scene into texture
class SceneTextureGL
{
public:
    // binds this framebuffer as main rendering target
    // clears it if `clear` flag is set
    void bind( bool clear );
    // binds default framebuffer (and read/draw framebuffers)
    void unbind();
    // if present-> del();gen();
    // otherwise just gen()
    // msaaPow - 2^msaaPow samples, msaaPow < 0 - use same default amount of samples
    void reset( const Vector2i& size, int msaaPow );
    // copy texture so draw() can render it
    void copyTexture();
    // renders texture
    void draw();
    // return true if texture is bound
    bool isBound() const { return isBound_; }
private:
    FramebufferData fd_;
    QuadTextureVertexObject qt_;
    bool isBound_{ false };
};

}