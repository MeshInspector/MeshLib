#pragma once
#ifdef __APPLE__

#include "MRGestureRecognizerHandler.h"

#include <AppKit/AppKit.h>

namespace MR
{

class GestureRecognizerCocoaHandler : public GestureRecognizerHandler::Impl
{
public:
    explicit GestureRecognizerCocoaHandler( GLFWwindow* window );
    ~GestureRecognizerCocoaHandler() override;

    GestureRecognizerHandler::MagnificationCallback magnificationCb;
    GestureRecognizerHandler::RotationCallback rotationCb;
    GestureRecognizerHandler::ScrollCallback mouseScrollCb;
    GestureRecognizerHandler::ScrollCallback touchScrollCb;

    void onMagnification( GestureRecognizerHandler::MagnificationCallback cb ) override;
    void onRotation( GestureRecognizerHandler::RotationCallback cb ) override;
    void onMouseScroll( GestureRecognizerHandler::ScrollCallback cb ) override;
    void onTouchScroll( GestureRecognizerHandler::ScrollCallback cb ) override;

private:
    NSView* view_;

    NSMagnificationGestureRecognizer* magnificationGestureRecognizer_;
    NSRotationGestureRecognizer* rotationGestureRecognizer_;
};

}

#endif