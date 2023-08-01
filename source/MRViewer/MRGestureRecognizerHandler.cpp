#include "MRGestureRecognizerHandler.h"
#include "MRGestureRecognizerCocoaHandler.h"

namespace MR
{
    void GestureRecognizerHandler::initialize( GLFWwindow* window )
    {
#ifdef __APPLE__
        impl_ = std::make_unique<GestureRecognizerCocoaHandler>( window );
#endif
    }

    void GestureRecognizerHandler::onRotation( GestureRecognizerHandler::RotationCallback cb )
    {
        if ( impl_ )
        {
            impl_->onRotation( cb );
        }
    }
}