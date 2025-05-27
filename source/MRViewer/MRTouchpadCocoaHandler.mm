#ifdef __APPLE__

#include "MRTouchpadCocoaHandler.h"
#include "MRViewer.h"

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

#include <AppKit/AppKit.h>

#include <objc/objc-runtime.h>

#include <atomic>
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

class TouchpadCocoaHandler::Impl
{
public:
    Impl( GLFWwindow* window, TouchpadCocoaHandler* handler );
    ~Impl();

    static void onMagnificationGestureEvent( NSView* view, SEL cmd, NSMagnificationGestureRecognizer* recognizer );
    static void onRotationGestureEvent( NSView* view, SEL cmd, NSRotationGestureRecognizer* recognizer );
    static void onScrollEvent( NSView* view, SEL cmd, NSEvent* event );

private:
    NSView* view_;

    NSMagnificationGestureRecognizer* magnificationGestureRecognizer_;
    NSRotationGestureRecognizer* rotationGestureRecognizer_;
    IMP previousScrollWheelMethod_;
};

TouchpadCocoaHandler::TouchpadCocoaHandler( GLFWwindow* window )
    : impl_( std::make_unique<Impl>( window, this ) )
{
    //
}

TouchpadCocoaHandler::~TouchpadCocoaHandler()
{
    //
}

TouchpadCocoaHandler::Impl::Impl(GLFWwindow *window, TouchpadCocoaHandler *handler)
    : view_( ( (NSWindow*)glfwGetCocoaWindow( window ) ).contentView )
{
    Class cls = [view_ class];

    magnificationGestureRecognizer_ = [[NSMagnificationGestureRecognizer alloc] initWithTarget:view_ action:@selector(handleMagnificationGesture:)];
    if ( !class_respondsToSelector( cls, @selector(handleMagnificationGesture:) ) )
        class_addMethod( cls, @selector(handleMagnificationGesture:), (IMP)Impl::onMagnificationGestureEvent, "v@:@" );
    [view_ addGestureRecognizer:magnificationGestureRecognizer_];

    rotationGestureRecognizer_ = [[NSRotationGestureRecognizer alloc] initWithTarget:view_ action:@selector(handleRotationGesture:)];
    if ( !class_respondsToSelector( cls, @selector(handleRotationGesture:) ) )
        class_addMethod( cls, @selector(handleRotationGesture:), (IMP)Impl::onRotationGestureEvent, "v@:@" );
    [view_ addGestureRecognizer:rotationGestureRecognizer_];

    // NOTE: GLFW scroll handler is replaced here
    if ( !class_respondsToSelector( cls, @selector(scrollWheel:) ) )
    {
        previousScrollWheelMethod_ = nil;
        class_addMethod( cls, @selector(scrollWheel:), (IMP)Impl::onScrollEvent, "v@:@" );
    }
    else
    {
        previousScrollWheelMethod_ = (IMP)[view_ methodForSelector:@selector(scrollWheel:)];
        class_replaceMethod( cls, @selector(scrollWheel:), (IMP)Impl::onScrollEvent, "v@:@" );
    }

    TouchpadCocoaHandlerRegistry::instance().add( view_, handler );
}

TouchpadCocoaHandler::Impl::~Impl()
{
    if ( previousScrollWheelMethod_ != nil )
    {
        Class cls = [view_ class];
        class_replaceMethod( cls, @selector(scrollWheel:), (IMP)previousScrollWheelMethod_, "v@:@" );
    }
    [rotationGestureRecognizer_ release];
    [magnificationGestureRecognizer_ release];
}

void TouchpadCocoaHandler::Impl::onMagnificationGestureEvent( NSView* view, SEL, NSMagnificationGestureRecognizer* recognizer )
{
    auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
    if ( !handler )
        return;

    const auto state = convert( recognizer.state );
    if ( state )
        handler->zoom( 1.f + recognizer.magnification, false, *state );
}

void TouchpadCocoaHandler::Impl::onRotationGestureEvent( NSView* view, SEL, NSRotationGestureRecognizer* recognizer )
{
    auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
    if ( !handler )
        return;

    const auto state = convert( recognizer.state );
    if ( state )
        handler->rotate( recognizer.rotation, *state );
}

void TouchpadCocoaHandler::Impl::onScrollEvent( NSView* view, SEL, NSEvent* event )
{
    auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
    if ( !handler )
        return;

    auto deltaX = [event scrollingDeltaX];
    auto deltaY = [event scrollingDeltaY];
    NSEventPhase phase = [event phase];

    if (
        [event subtype] == NSEventSubtypeMouseEvent ||
        // We know exactly one Mac machine where mouse scroll events arrive with this subtype. Some sort of a bug?
        // We also have to filter by `phase == NSEventPhaseNone` here, because otherwise this incorrectly catches the "move two fingers in any direciton" event from the touchpad (!!),
        //   which is instead supposed to rotate the camera, not act as a scroll.
        // And ALSO we have to filter by `[event momentumPhase] == NSEventPhaseNone`, because when you release that two-finger gesture while moving the fingers (so as to
        //   trigger kinetic movement), the kinetic part of the gesture would be caught here and interpreted as a scroll, and we don't want that. We want it to continue the rotation movement instead.
        ([event subtype] == NSEventSubtypeApplicationActivated && phase == NSEventPhaseNone && [event momentumPhase] == NSEventPhaseNone)
    )
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
        std::optional<GestureState> state = std::nullopt;
        bool kinetic = false;
        if ( ( state = convert( phase ) ) )
            kinetic = false;
        else if ( ( state = convert( [event momentumPhase] ) ) )
            kinetic = true;
        else
            return;

        // merge consecutive swipe gestures
        static std::atomic_bool gDelayedSwipeGestureEnd = false;
        if ( *state == GestureState::Begin && gDelayedSwipeGestureEnd.exchange( false ) )
        {
            return;
        }
        else if ( *state == GestureState::End )
        {
            gDelayedSwipeGestureEnd.store( true );
            constexpr auto delayMs = 50.;
            auto popTime = dispatch_time( DISPATCH_TIME_NOW, (int64_t)( delayMs * NSEC_PER_MSEC ) );
            dispatch_after( popTime, dispatch_get_main_queue(), ^(void)
            {
                if ( gDelayedSwipeGestureEnd.exchange( false ) )
                {
                    handler->swipe( deltaX, deltaY, false, GestureState::End );
                    // manually resume the event loop
                    getViewerInstance().postEmptyEvent();
                }
            } );
        }
        else
        {
            handler->swipe( deltaX, deltaY, kinetic, *state );
        }
    }
}

}

#endif