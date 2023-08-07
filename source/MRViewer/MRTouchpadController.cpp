#include "MRTouchpadController.h"
#include "MRTouchpadCocoaHandler.h"

namespace MR
{
    void TouchpadController::initialize( GLFWwindow* window )
    {
#ifdef __APPLE__
        impl_ = std::make_unique<TouchpadCocoaHandler>( this, window );
#endif
    }

    void TouchpadController::onMagnification( TouchpadController::MagnificationCallback cb )
    {
        magnificationCb_ = std::move( cb );
    }

    void TouchpadController::onRotation( TouchpadController::RotationCallback cb )
    {
        rotationCb_ = std::move( cb );
    }

    void TouchpadController::onMouseScroll( TouchpadController::ScrollCallback cb )
    {
        mouseScrollCb_ = std::move( cb );
    }

    void TouchpadController::onTouchScroll( TouchpadController::ScrollCallback cb )
    {
        touchScrollCb_ = std::move( cb );
    }

    TouchpadController::Impl::Impl( TouchpadController* controller, GLFWwindow* )
        : controller_( controller )
    {
        //
    }

    void TouchpadController::Impl::mouseScroll( float dx, float dy )
    {
        controller_->mouseScrollCb_( dx, dy );
    }

    void TouchpadController::Impl::rotate( float angle, bool finished )
    {
        controller_->rotationCb_( angle, finished );
    }

    void TouchpadController::Impl::swipe( float dx, float dy )
    {
        controller_->touchScrollCb_( dx, dy );
    }

    void TouchpadController::Impl::zoom( float scale, bool finished )
    {
        controller_->magnificationCb_( scale, finished );
    }
}