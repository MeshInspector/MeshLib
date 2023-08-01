#ifdef __APPLE__

#include "MRGestureRecognizerCocoaHandler.h"

#include <MRMesh/MRVector2.h>

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

#include <objc/objc-runtime.h>

#include <map>
#import <spdlog/spdlog.h>

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
            handler->magnificationCb( magnificationGestureRecognizer.magnification );
        }
    }

    void panGestureEvent( NSView* view, SEL cmd, NSPanGestureRecognizer* panGestureRecognizer )
    {
        auto* handler = GestureRecognizerCocoaHandlerRegistry::instance().find( view );
        if ( handler )
        {
            auto translation = [panGestureRecognizer translationInView:view];
            MR::Vector2f p1( translation.x, translation.y );
            spdlog::info( "pan gesture ( button mask = {} )", panGestureRecognizer.buttonMask );
        }
    }

    void rotationGestureEvent( NSView* view, SEL cmd, NSRotationGestureRecognizer* rotationGestureRecognizer )
    {
        auto* handler = GestureRecognizerCocoaHandlerRegistry::instance().find( view );
        if ( handler )
        {
            handler->rotationCb( rotationGestureRecognizer.rotation );
        }
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

        panGestureRecognizer_ = [[NSPanGestureRecognizer alloc] initWithTarget:view_ action:@selector(handlePanGesture:)];
        if ( !class_respondsToSelector( cls, @selector(handlePanGesture:) ) )
            class_addMethod( cls, @selector(handlePanGesture:), (IMP)panGestureEvent, "v@:@" );
        [view_ addGestureRecognizer:panGestureRecognizer_];

        rotationGestureRecognizer_ = [[NSRotationGestureRecognizer alloc] initWithTarget:view_ action:@selector(handleRotationGesture:)];
        if ( !class_respondsToSelector( cls, @selector(handleRotationGesture:) ) )
            class_addMethod( cls, @selector(handleRotationGesture:), (IMP)rotationGestureEvent, "v@:@" );
        [view_ addGestureRecognizer:rotationGestureRecognizer_];

        GestureRecognizerCocoaHandlerRegistry::instance().add( view_, this );
    }

    GestureRecognizerCocoaHandler::~GestureRecognizerCocoaHandler()
    {
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

