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

    void onTouchesBegan( NSView* view, SEL cmd, NSEvent* event )
    {
        auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
        if ( ! handler )
            return;
        if ( !handler->touchCb )
            return;

        NSSet* touches = [event touchesMatchingPhase:NSTouchPhaseBegan inView:view];
        NSArray* array = [touches allObjects];
        for ( auto i = 0; i < [array count]; ++i )
        {
            NSTouch* touch = [array objectAtIndex:i];
            auto id = touch.identity;
            auto pos = touch.normalizedPosition;
            handler->touchCb( id.hash, pos.x, pos.y, MR::TouchpadController::TouchState::Began );
        }
    }

    void onTouchesMoved( NSView* view, SEL cmd, NSEvent* event )
    {
        auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
        if ( ! handler )
            return;
        if ( !handler->touchCb )
            return;

        NSSet* touches = [event touchesMatchingPhase:NSTouchPhaseMoved inView:view];
        NSArray* array = [touches allObjects];
        for ( auto i = 0; i < [array count]; ++i )
        {
            NSTouch* touch = [array objectAtIndex:i];
            auto id = touch.identity;
            auto pos = touch.normalizedPosition;
            handler->touchCb( id.hash, pos.x, pos.y, MR::TouchpadController::TouchState::Moved );
        }
    }

    void onTouchesEnded( NSView* view, SEL cmd, NSEvent* event )
    {
        auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
        if ( ! handler )
            return;
        if ( !handler->touchCb )
            return;

        NSSet* touches = [event touchesMatchingPhase:NSTouchPhaseMoved inView:view];
        NSArray* array = [touches allObjects];
        for ( auto i = 0; i < [array count]; ++i )
        {
            NSTouch* touch = [array objectAtIndex:i];
            auto id = touch.identity;
            auto pos = touch.normalizedPosition;
            handler->touchCb( id.hash, pos.x, pos.y, MR::TouchpadController::TouchState::Ended );
        }
    }

    void onTouchesCancelled( NSView* view, SEL cmd, NSEvent* event )
    {
        auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
        if ( ! handler )
            return;
        if ( !handler->touchCb )
            return;

        NSSet* touches = [event touchesMatchingPhase:NSTouchPhaseCancelled inView:view];
        NSArray* array = [touches allObjects];
        for ( auto i = 0; i < [array count]; ++i )
        {
            NSTouch* touch = [array objectAtIndex:i];
            auto id = touch.identity;
            auto pos = touch.normalizedPosition;
            handler->touchCb( id.hash, pos.x, pos.y, MR::TouchpadController::TouchState::Canceled );
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

        [view_ setAllowedTouchTypes:(NSTouchTypeMaskDirect | NSTouchTypeMaskIndirect)];
        // FIXME: find where the methods were defined previously
        //if ( !class_respondsToSelector( cls, @selector(touchesBeganWithEvent:) ) )
            class_addMethod( cls, @selector(touchesBeganWithEvent:), (IMP)onTouchesBegan, "v@:@" );
        //if ( !class_respondsToSelector( cls, @selector(touchesMovedWithEvent:) ) )
            class_addMethod( cls, @selector(touchesMovedWithEvent:), (IMP)onTouchesMoved, "v@:@" );
        //if ( !class_respondsToSelector( cls, @selector(touchesEndedWithEvent:) ) )
            class_addMethod( cls, @selector(touchesEndedWithEvent:), (IMP)onTouchesEnded, "v@:@" );
        //if ( !class_respondsToSelector( cls, @selector(touchesCanceledWithEvent:) ) )
            class_addMethod( cls, @selector(touchesCancelledWithEvent:), (IMP)onTouchesCancelled, "v@:@" );

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

    void TouchpadCocoaHandler::onTouch( TouchpadController::TouchCallback cb )
    {
        // TODO: thread safety?
        touchCb = cb;
    }
}

#endif