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

    GestureRecognizerHandler::RotationCallback rotationCb;

    void onRotation( GestureRecognizerHandler::RotationCallback cb ) override;

private:
    NSView* view_;

    NSRotationGestureRecognizer* rotationGestureRecognizer_;
};

}

#endif