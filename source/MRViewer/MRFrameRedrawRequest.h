#pragma once
#include "exports.h"
#include "MRAsyncTimer.h"

namespace MR
{

/// class for requesting frame redraw in some time
class MRVIEWER_CLASS FrameRedrawRequest
{
public:
    MRVIEWER_API void reset();
    MRVIEWER_API void requestFrame( size_t millisecondsInterval = 100 );
private:
#ifdef __EMSCRIPTEN__
    std::atomic<bool> frameRequested_{ false }; // not to order too often
#else
    AsyncRequest asyncRequest_;
#endif
};

}