#ifdef __APPLE__

#include "MRTouchpadCocoaHandler.h"

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

std::optional<MR::TouchpadController::Handler::GestureState> convert( NSGestureRecognizerState state )
{
    using GS = MR::TouchpadController::Handler::GestureState;
    switch ( state )
    {
        case NSGestureRecognizerStateBegan:
            return GS::Begin;
        case NSGestureRecognizerStateChanged:
            return GS::Update;
        case NSGestureRecognizerStateEnded:
        case NSGestureRecognizerStateCancelled:
            return GS::End;
        default:
            return std::nullopt;
    }
}

std::optional<MR::TouchpadController::Handler::GestureState> convert( NSEventPhase phase )
{
    using GS = MR::TouchpadController::Handler::GestureState;
    switch ( phase )
    {
        case NSEventPhaseNone:
        case NSEventPhaseMayBegin:
        case NSEventPhaseStationary:
            return std::nullopt;
        case NSEventPhaseBegan:
            return GS::Begin;
        case NSEventPhaseChanged:
            return GS::Update;
        case NSEventPhaseEnded:
        case NSEventPhaseCancelled:
            return GS::End;
        default:
            return std::nullopt;
    }
}

}

namespace MR
{

TouchpadCocoaHandler::TouchpadCocoaHandler( GLFWwindow* window )
    : view_( ( (NSWindow*)glfwGetCocoaWindow( window ) ).contentView )
{
    Class cls = [view_ class];

    magnificationGestureRecognizer_ = [[NSMagnificationGestureRecognizer alloc] initWithTarget:view_ action:@selector(handleMagnificationGesture:)];
    if ( !class_respondsToSelector( cls, @selector(handleMagnificationGesture:) ) )
        class_addMethod( cls, @selector(handleMagnificationGesture:), (IMP)TouchpadCocoaHandler::onMagnificationGestureEvent, "v@:@" );
    [view_ addGestureRecognizer:magnificationGestureRecognizer_];

    rotationGestureRecognizer_ = [[NSRotationGestureRecognizer alloc] initWithTarget:view_ action:@selector(handleRotationGesture:)];
    if ( !class_respondsToSelector( cls, @selector(handleRotationGesture:) ) )
        class_addMethod( cls, @selector(handleRotationGesture:), (IMP)TouchpadCocoaHandler::onRotationGestureEvent, "v@:@" );
    [view_ addGestureRecognizer:rotationGestureRecognizer_];

    // NOTE: GLFW scroll handler is replaced here
    if ( !class_respondsToSelector( cls, @selector(scrollWheel:) ) )
    {
        previousScrollWheelMethod_ = nil;
        class_addMethod( cls, @selector(scrollWheel:), (IMP)TouchpadCocoaHandler::onScrollEvent, "v@:@" );
    }
    else
    {
        previousScrollWheelMethod_ = (IMP)[view_ methodForSelector:@selector(scrollWheel:)];
        class_replaceMethod( cls, @selector(scrollWheel:), (IMP)TouchpadCocoaHandler::onScrollEvent, "v@:@" );
    }

    TouchpadCocoaHandlerRegistry::instance().add( view_, this );
}

TouchpadCocoaHandler::~TouchpadCocoaHandler()
{
    if ( previousScrollWheelMethod_ != nil )
    {
        Class cls = [view_ class];
        class_replaceMethod( cls, @selector(scrollWheel:), (IMP)previousScrollWheelMethod_, "v@:@" );
    }
    [rotationGestureRecognizer_ release];
    [magnificationGestureRecognizer_ release];
}

void TouchpadCocoaHandler::onMagnificationGestureEvent( NSView* view, SEL cmd, NSMagnificationGestureRecognizer* recognizer )
{
    auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
    if ( !handler )
        return;

    const auto state = convert( recognizer.state );
    if ( state )
        handler->zoom( 1.f + recognizer.magnification, false, *state );
}

void TouchpadCocoaHandler::onRotationGestureEvent( NSView* view, SEL cmd, NSRotationGestureRecognizer* recognizer )
{
    auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
    if ( !handler )
        return;

    const auto state = convert( recognizer.state );
    if ( state )
        handler->rotate( recognizer.rotation, *state );
}

void TouchpadCocoaHandler::onScrollEvent( NSView* view, SEL cmd, NSEvent* event )
{
    auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
    if ( !handler )
        return;

    auto deltaX = [event scrollingDeltaX];
    auto deltaY = [event scrollingDeltaY];
    if ( [event subtype] == NSEventSubtypeMouseEvent )
    {
        if ( deltaX == 0.0 && deltaY == 0.0 )
        {
            return;
        }
        if ( [event hasPreciseScrollingDeltas] )
        {
            deltaX *= 0.1;
            deltaY *= 0.1;
        }
        handler->mouseScroll( deltaX, deltaY, [event momentumPhase] != NSEventPhaseNone );
    }
    else
    {
        if ( const auto state = convert( [event phase] ) )
            handler->swipe( deltaX, deltaY, false, *state );
        else if ( const auto momentumPhase = convert( [event momentumPhase] ) )
            handler->swipe( deltaX, deltaY, true, *momentumPhase );
    }
}

}

#endif