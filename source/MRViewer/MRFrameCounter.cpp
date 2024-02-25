#include "MRFrameCounter.h"

namespace MR
{

void FrameCounter::endDraw( bool swapped )
{
    ++totalFrameCounter;
    if ( swapped )
    {
        ++swappedFrameCounter;
        const auto nowTP = std::chrono::high_resolution_clock::now();
        const auto nowSec = std::chrono::time_point_cast<std::chrono::seconds>( nowTP ).time_since_epoch().count();
        drawTimeMilliSec =  ( nowTP - startDrawTime_ ) * 1000;
        if ( nowSec > startFPSTime_ )
        {
            startFPSTime_ = nowSec;
            fps = swappedFrameCounter - startFrameNum;
            startFrameNum = swappedFrameCounter;
        }
    }
}

void FrameCounter::reset()
{
    totalFrameCounter = 0;
    swappedFrameCounter = 0;
    startFPSTime_ = 0;
    fps = 0;
    startFrameNum = 0;
}

} //namespace MR
