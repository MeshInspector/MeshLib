#ifdef __APPLE__

#include "MRGestureRecognizerCocoaHandler.h"

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

#include <objc/objc-runtime.h>

#include <map>

#include <spdlog/spdlog.h>

namespace
{
    class GestureRecognizerCocoaHandlerRegistry
    {
    public:
        static GestureRecognizerCocoaHandlerRegistry& instance()
        {
            static GestureRecognizerCocoaHandlerRegistry instance;
            return instance;
        }

        void add( NSView* view, MR::GestureRecognizerCocoaHandler* handler )
        {
            registry_.emplace( view, handler );
        }

        void remove( NSView* view )
        {
            registry_.erase( view );
        }

        [[nodiscard]] MR::GestureRecognizerCocoaHandler* find( NSView* view ) const
        {
            const auto it = registry_.find( view );
            if ( it != registry_.end() )
                return it->second;
            else
                return nullptr;
        }

    private:
        std::map<NSView*, MR::GestureRecognizerCocoaHandler*> registry_;
    };

    void magnificationGestureEvent( NSView* view, SEL cmd, NSMagnificationGestureRecognizer* magnificationGestureRecognizer )
    {
        auto* handler = GestureRecognizerCocoaHandlerRegistry::instance().find( view );
        if ( handler )
        {
            const auto finished = magnificationGestureRecognizer.state == NSGestureRecognizerStateEnded;
            handler->magnificationCb( magnificationGestureRecognizer.magnification, finished );
        }
    }

    void rotationGestureEvent( NSView* view, SEL cmd, NSRotationGestureRecognizer* rotationGestureRecognizer )
    {
        auto* handler = GestureRecognizerCocoaHandlerRegistry::instance().find( view );
        if ( handler )
        {
            const auto finished = rotationGestureRecognizer.state == NSGestureRecognizerStateEnded;
            handler->rotationCb( rotationGestureRecognizer.rotation, finished );
        }
    }

    void onTouchesBegan( NSView* view, SEL cmd, NSEvent* event )
    {
        spdlog::info( "touches began" );
    }

    void onTouchesMoved( NSView* view, SEL cmd, NSEvent* event )
    {
        spdlog::info( "touches moved" );
    }

    void onTouchesEnded( NSView* view, SEL cmd, NSEvent* event )
    {
        spdlog::info( "touches ended" );
    }
}

namespace MR
{
    GestureRecognizerCocoaHandler::GestureRecognizerCocoaHandler( GLFWwindow* window )
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

        [view_ setAllowedTouchTypes:(NSTouchTypeMaskDirect | NSTouchTypeMaskIndirect)];
        // FIXME: find where the methods were defined previously
        //if ( !class_respondsToSelector( cls, @selector(touchesBeganWithEvent:) ) )
            class_addMethod( cls, @selector(touchesBeganWithEvent:), (IMP)onTouchesBegan, "v@:@" );
        //if ( !class_respondsToSelector( cls, @selector(touchesMovedWithEvent:) ) )
            class_addMethod( cls, @selector(touchesMovedWithEvent:), (IMP)onTouchesMoved, "v@:@" );
        //if ( !class_respondsToSelector( cls, @selector(touchesEndedWithEvent:) ) )
            class_addMethod( cls, @selector(touchesEndedWithEvent:), (IMP)onTouchesEnded, "v@:@" );

        GestureRecognizerCocoaHandlerRegistry::instance().add( view_, this );
    }

    GestureRecognizerCocoaHandler::~GestureRecognizerCocoaHandler()
    {
        [magnificationGestureRecognizer_ release];
        [rotationGestureRecognizer_ release];
    }

    void GestureRecognizerCocoaHandler::onMagnification( GestureRecognizerHandler::MagnificationCallback cb )
    {
        // TODO: thread safety?
        magnificationCb = cb;
    }

    void GestureRecognizerCocoaHandler::onRotation( GestureRecognizerHandler::RotationCallback cb )
    {
        // TODO: thread safety?
        rotationCb = cb;
    }
}

#endif

