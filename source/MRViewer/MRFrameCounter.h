#pragma once

#include "MRViewerFwd.h"
#include <chrono>

namespace MR
{

class FrameCounter
{
public:
    size_t totalFrameCounter{ 0 };
    size_t swappedFrameCounter{ 0 };
    size_t startFrameNum{ 0 };
    size_t fps{ 0 };
    std::chrono::duration<double> drawTimeMilliSec{ 0 };

    void startDraw() { startDrawTime_ = std::chrono::high_resolution_clock::now(); }
    MRVIEWER_API void endDraw( bool swapped );
    MRVIEWER_API void reset();

private:
    long long startFPSTime_{ 0 };
    std::chrono::time_point<std::chrono::high_resolution_clock> startDrawTime_;
};

} //namespace MR
