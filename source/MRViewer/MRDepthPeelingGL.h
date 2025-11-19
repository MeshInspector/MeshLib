#pragma once
#include "MRViewerFwd.h"
#include "MRRenderGLHelpers.h"

namespace MR
{

/// class to encapsulate depth peeling rendering passes as fall back if alpha sort is not available 
class MRVIEWER_CLASS DepthPeelingGL
{
public:
    MR_ADD_CTOR_DELETE_MOVE( DepthPeelingGL );

    /// if present-> del();gen();
    /// otherwise just gen()
    void reset( const Vector2i& size );

    /// renders transparent objects into this buffer
    /// returns true if there are transparent objects to render
    bool doPasses( FramebufferData* bgFramebuffer );

    /// draws this result texture onto
    void draw();
private:
    FramebufferData accumFB_;
    QuadTextureVertexObject qt_;
    const int numPasses_{ 4 };
};

}