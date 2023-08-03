#include "MRTouchpadController.h"
#include "MRTouchpadCocoaHandler.h"

namespace MR
{
    void TouchpadController::initialize( GLFWwindow* window )
    {
#ifdef __APPLE__
        impl_ = std::make_unique<TouchpadCocoaHandler>( window );
#endif
    }

    void TouchpadController::onMagnification( TouchpadController::MagnificationCallback cb )
    {
        if ( impl_ )
        {
            impl_->onMagnification( cb );
        }
    }

    void TouchpadController::onRotation( TouchpadController::RotationCallback cb )
    {
        if ( impl_ )
        {
            impl_->onRotation( cb );
        }
    }

    void TouchpadController::onMouseScroll( TouchpadController::ScrollCallback cb )
    {
        if ( impl_ )
        {
            impl_->onMouseScroll( cb );
        }
    }

    void TouchpadController::onTouchScroll( TouchpadController::ScrollCallback cb )
    {
        if ( impl_ )
        {
            impl_->onTouchScroll( cb );
        }
    }
}