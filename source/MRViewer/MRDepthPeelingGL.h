#pragma once
#include "MRViewerFwd.h"
#include "MRRenderGLHelpers.h"
#include <array>

namespace MR
{

/// class to encapsulate depth peeling rendering passes as fall back if alpha sort is not available 
class MRVIEWER_CLASS DepthPeelingGL
{
public:
    MR_ADD_CTOR_DELETE_MOVE( DepthPeelingGL );

    /// if present-> del();gen();
    /// otherwise just gen()
    MRVIEWER_API void reset( const Vector2i& size );

    /// renders transparent objects into this buffer
    /// returns true if there are transparent objects to render
    MRVIEWER_API bool doPasses( FramebufferData* bgFramebuffer );

    /// draws this result texture onto
    MRVIEWER_API void draw();

    /// functions to control number of passes
    int getNumPasses() const { return numPasses_; }
    void setNumPasses( int passes ) { numPasses_ = passes; }
private:
    // draws a full-screen quad with DepthPeelQuad shader
    void drawQuad_( unsigned colorTex, unsigned depthTex, bool dilate );

    // number of the nearest layers composited separately, each exact on every MSAA sample
    static constexpr int cSeparateLayers = 2;

    struct Target
    {
        unsigned fbo = 0;
        GlTexture2 color;
        GlTexture2 depth;
    };
    // [0, cSeparateLayers): the nearest layers, each alone;
    // [cSeparateLayers]: the next layer, the farther ones are accumulated behind it in accumFB_;
    // [cSeparateLayers + 1]: the farthest opaque depth around each pixel
    std::array<Target, cSeparateLayers + 2> targets_;

    FramebufferData accumFB_;
    QuadTextureVertexObject qt_;
    int numPasses_{ 5 };
};

}