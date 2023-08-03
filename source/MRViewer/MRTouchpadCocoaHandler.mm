#ifdef __APPLE__

#include "MRGestureRecognizerCocoaHandler.h"

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

#include <objc/objc-runtime.h>

#include <map>

namespace
{
    class TouchpadCocoaHandlerRegistry
    {
    public:
        static TouchpadCocoaHandlerRegistry& instance()
        {
            static TouchpadCocoaHandlerRegistry instance;
            return instance;
        }

        void add( NSView* view, MR::TouchpadCocoaHandler* handler )
        {
            registry_.emplace( view, handler );
        }

        void remove( NSView* view )
        {
            registry_.erase( view );
        }

        [[nodiscard]] MR::TouchpadCocoaHandler* find( NSView* view ) const
        {
            const auto it = registry_.find( view );
            if ( it != registry_.end() )
                return it->second;
            else
                return nullptr;
        }

    private:
        std::map<NSView*, MR::TouchpadCocoaHandler*> registry_;
    };

    void magnificationGestureEvent( NSView* view, SEL cmd, NSMagnificationGestureRecognizer* magnificationGestureRecognizer )
    {
        auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
        if ( !handler )
            return;
        if ( !handler->magnificationCb )
            return;

        const auto finished = magnificationGestureRecognizer.state == NSGestureRecognizerStateEnded;
        handler->magnificationCb( std::exp( -magnificationGestureRecognizer.magnification ), finished );
    }

    void rotationGestureEvent( NSView* view, SEL cmd, NSRotationGestureRecognizer* rotationGestureRecognizer )
    {
        auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
        if ( ! handler )
            return;
        if ( !handler->rotationCb )
            return;

        const auto finished = rotationGestureRecognizer.state == NSGestureRecognizerStateEnded;
        handler->rotationCb( rotationGestureRecognizer.rotation, finished );
    }

    void scrollEvent( NSView* view, SEL cmd, NSEvent* event )
    {
        auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
        if ( !handler )
            return;

        auto deltaX = [event scrollingDeltaX];
        auto deltaY = [event scrollingDeltaY];
        if ( [event hasPreciseScrollingDeltas] )
        {
            deltaX *= 0.1;
            deltaY *= 0.1;
        }
        if ( deltaX == 0.0 && deltaY == 0.0 )
            return;

        if ( [event subtype] == NSEventSubtypeMouseEvent )
        {
            if ( handler->mouseScrollCb )
                handler->mouseScrollCb( deltaX, deltaY );
        }
        else
        {
            if ( handler->touchScrollCb )
                handler->touchScrollCb( deltaX, deltaY );
        }
    }
}

namespace MR
{
    TouchpadCocoaHandler::TouchpadCocoaHandler( GLFWwindow* window )
    {
        auto* nsWindow = (NSWindow*)glfwGetCocoaWindow( window );
        view_ = nsWindow.contentView;

        Class cls = [view_ class];

        magnificationGestureRecognizer_ = [[NSMagnificationGestureRecognizer alloc] initWithTarget:view_ action:@selector(handleMagnificationGesture:)];
        if ( !class_respondsToSelector( cls, @selector(handleMagnificationGesture:) ) )
            class_addMethod( cls, @selector(handleMagnificationGesture:), (IMP)magnificationGestureEvent, "v@:@" );
        [view_ addGestureRecognizer:magnificationGestureRecognizer_];

        rotationGestureRecognizer_ = [[NSRotationGestureRecognizer alloc] initWithTarget:view_ action:@selector(handleRotationGesture:)];
        if ( !class_respondsToSelector( cls, @selector(handleRotationGesture:) ) )
            class_addMethod( cls, @selector(handleRotationGesture:), (IMP)rotationGestureEvent, "v@:@" );
        [view_ addGestureRecognizer:rotationGestureRecognizer_];

        // NOTE: GLFW scroll handler is replaced here
        if ( !class_respondsToSelector( cls, @selector(scrollWheel:) ) )
            class_addMethod( cls, @selector(scrollWheel:), (IMP)scrollEvent, "v@:@" );
        else
            class_replaceMethod( cls, @selector(scrollWheel:), (IMP)scrollEvent, "v@:@" );

        TouchpadCocoaHandlerRegistry::instance().add( view_, this );
    }

    TouchpadCocoaHandler::~TouchpadCocoaHandler()
    {
        [magnificationGestureRecognizer_ release];
        [rotationGestureRecognizer_ release];
    }

    void TouchpadCocoaHandler::onMagnification( TouchpadController::MagnificationCallback cb )
    {
        // TODO: thread safety?
        magnificationCb = cb;
    }

    void TouchpadCocoaHandler::onRotation( TouchpadController::RotationCallback cb )
    {
        // TODO: thread safety?
        rotationCb = cb;
    }

    void TouchpadCocoaHandler::onMouseScroll( TouchpadController::ScrollCallback cb )
    {
        // TODO: thread safety?
        mouseScrollCb = cb;
    }

    void TouchpadCocoaHandler::onTouchScroll( TouchpadController::ScrollCallback cb )
    {
        // TODO: thread safety?
        touchScrollCb = cb;
    }
}

#endif