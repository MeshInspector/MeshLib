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

    void TouchpadController::onZoom( ZoomCallback cb )
    {
        zoomCb_ = std::move( cb );
    }

    void TouchpadController::onRotate( RotateCallback cb )
    {
        rotateCb_ = std::move( cb );
    }

    void TouchpadController::onMouseScroll( ScrollSwipeCallback cb )
    {
        mouseScrollCb_ = std::move( cb );
    }

    void TouchpadController::onSwipe( ScrollSwipeCallback cb )
    {
        swipeCb_ = std::move( cb );
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

    void TouchpadController::Impl::rotate( float angle, GestureState state )
    {
        controller_->rotateCb_( angle, state );
    }

    void TouchpadController::Impl::swipe( float dx, float dy )
    {
        controller_->swipeCb_( dx, dy );
    }

    void TouchpadController::Impl::zoom( float scale, GestureState state )
    {
        controller_->zoomCb_( scale, state );
    }
}