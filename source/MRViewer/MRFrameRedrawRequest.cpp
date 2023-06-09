#include "MRFrameRedrawRequest.h"
#include "MRViewer.h"
#include "MRPch/MRWasm.h"
#include <chrono>

namespace MR
{

void FrameRedrawRequest::reset()
{
#ifdef __EMSCRIPTEN__
    bool testFrameOrder = true;
    frameRequested_.compare_exchange_strong( testFrameOrder, false );
#endif
}

void FrameRedrawRequest::requestFrame( size_t millisecondsInterval )
{
    // do not do it too frequently not to overload the renderer
    auto minInterval = std::chrono::milliseconds( millisecondsInterval );
    // make request
#ifdef __EMSCRIPTEN__
    bool testFrameOrder = false;
    if ( frameRequested_.compare_exchange_strong( testFrameOrder, true ) )
    {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
        MAIN_THREAD_EM_ASM( postEmptyEvent( $0, 2 ), int( minInterval.count() ) );
#pragma clang diagnostic pop
    }
#else
    asyncRequest_.requestIfNotSet( std::chrono::system_clock::now() + minInterval, [] ()
    {
        getViewerInstance().postEmptyEvent();
    } );
#endif
}

}