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
    FramebufferData accumFB_;
    QuadTextureVertexObject qt_;
    GlTexture2 nearestDepth_; // depth of the nearest layer, `draw` is depth-tested with it
    unsigned nearestDepthFB_ = 0; // to copy the depth of accumFB_ after the first pass into nearestDepth_
    int numPasses_{ 5 };
};

}