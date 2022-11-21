#include "MRPch/MRWasm.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer.h"

#ifdef __EMSCRIPTEN__

//using namespace MR;

extern "C"
{
EMSCRIPTEN_KEEPALIVE int emsTouchStart( int id, float x, float y )
{
    return int( MR::getViewerInstance().touchStart( id, int(x), int(y) ) );
}

EMSCRIPTEN_KEEPALIVE int emsTouchEnd( int id, float x, float y )
{
    return int( MR::getViewerInstance().touchEnd( id, int(x), int(y) ) );
}

EMSCRIPTEN_KEEPALIVE int emsTouchMove( int id, float x, float y )
{
    return int( MR::getViewerInstance().touchMove( id, int(x), int(y) ) );
}

}
#endif