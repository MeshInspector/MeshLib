#pragma once

#include "MRViewerFwd.h"

namespace MR
{

struct TouchpadParameters
{
    /// most touchpads implement kinetic (or inertial) scrolling, this option disables handling of these events
    bool ignoreKineticMoves = false;
    /// enable gesture's cancellability, i.e. revert its changes in case of external interruption
    bool cancellable = false;
    /// swipe processing mode
    enum class SwipeMode {
        SwipeRotatesCamera = 0,
        SwipeMovesCamera = 1,
        Count
    } swipeMode = SwipeMode::SwipeRotatesCamera;
};

} //namespace MR
