#pragma once
#ifdef __APPLE__

#include "MRTouchpadController.h"

namespace MR
{

/// Touchpad event handler for macOS
class TouchpadCocoaHandler : public TouchpadController::Handler
{
public:
    explicit TouchpadCocoaHandler( GLFWwindow* window );
    ~TouchpadCocoaHandler() override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}

#endif