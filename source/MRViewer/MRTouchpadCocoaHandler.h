#pragma once
#ifdef __APPLE__

#include "MRGestureRecognizerHandler.h"

#include <AppKit/AppKit.h>

namespace MR
{

class TouchpadCocoaHandler : public TouchpadController::Impl
{
public:
    explicit TouchpadCocoaHandler( GLFWwindow* window );
    ~TouchpadCocoaHandler() override;

    TouchpadController::MagnificationCallback magnificationCb;
    TouchpadController::RotationCallback rotationCb;
    TouchpadController::ScrollCallback mouseScrollCb;
    TouchpadController::ScrollCallback touchScrollCb;

    void onMagnification( TouchpadController::MagnificationCallback cb ) override;
    void onRotation( TouchpadController::RotationCallback cb ) override;
    void onMouseScroll( TouchpadController::ScrollCallback cb ) override;
    void onTouchScroll( TouchpadController::ScrollCallback cb ) override;

private:
    NSView* view_;

    NSMagnificationGestureRecognizer* magnificationGestureRecognizer_;
    NSRotationGestureRecognizer* rotationGestureRecognizer_;
};

}

#endif