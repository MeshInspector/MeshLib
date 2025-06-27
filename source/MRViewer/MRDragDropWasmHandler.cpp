#ifdef __EMSCRIPTEN__
#include "MRViewer.h"
#include "MRPch/MRWasm.h"
#include "MRPch/MRSpdlog.h"

extern "C"
{

EMSCRIPTEN_KEEPALIVE void emsDragEnter()
{
    auto& v = MR::getViewerInstance();
    v.emplaceEvent( "Drag enter", [&v] ()
    {
        v.dragEntranceSignal( true );
    } );
}

EMSCRIPTEN_KEEPALIVE void emsDragLeave()
{
    auto& v = MR::getViewerInstance();
    v.emplaceEvent( "Drag leave", [&v] ()
    {
        v.dragEntranceSignal( false );
    } );
}


EMSCRIPTEN_KEEPALIVE void emsDragOver(int x, int y)
{
    auto& v = MR::getViewerInstance();
    v.emplaceEvent( "Drag over", [&v,x,y] () mutable
    {
        v.dragOverSignal( int( std::round( x * v.pixelRatio ) ), int( std::round( y * v.pixelRatio ) ) );
    }, true );
}

// drop event is produced by glfw

}

#endif