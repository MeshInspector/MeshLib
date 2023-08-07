#pragma once
#ifdef __APPLE__

#include "MRTouchpadController.h"

#include <AppKit/AppKit.h>

namespace MR
{

class TouchpadCocoaHandler : public TouchpadController::Impl
{
public:
    explicit TouchpadCocoaHandler( TouchpadController* controller, GLFWwindow* window );
    ~TouchpadCocoaHandler() override;

    static void onMagnificationGestureEvent( NSView* view, SEL cmd, NSMagnificationGestureRecognizer* recognizer );
    static void onRotationGestureEvent( NSView* view, SEL cmd, NSRotationGestureRecognizer* recognizer );
    static void onScrollEvent( NSView* view, SEL cmd, NSEvent* event );

private:
    NSView* view_;

    NSMagnificationGestureRecognizer* magnificationGestureRecognizer_;
    NSRotationGestureRecognizer* rotationGestureRecognizer_;
};

}

#endif