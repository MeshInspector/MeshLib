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
    /// msaaPow - 2^msaaPow samples, msaaPow < 0 - use same default amount of samples
    void reset( const Vector2i& size, int msaaPow );

    /// renders transparent objects into this buffer
    /// returns true if there is transparent objects to render
    bool doPasses( SceneTextureGL* sceneTexture );

    /// draws this result texture onto
    void draw();
private:
    FramebufferData accumFB_[2]; // do we need two? of them to swap
    QuadTextureVertexObject qt_;
    const int numPasses_{ 3 }; // bare minimum for test
};

}