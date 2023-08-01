#ifdef __APPLE__

#include "MRGestureRecognizerCocoaHandler.h"

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

#include <objc/objc-runtime.h>

#include <map>

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

        rotationGestureRecognizer_ = [[NSRotationGestureRecognizer alloc] initWithTarget:view_ action:@selector(handleRotateGesture:)];
        if ( !class_respondsToSelector( cls, @selector(handleRotateGesture:)) )
            class_addMethod( cls, @selector(handleRotateGesture:), (IMP)rotationGestureEvent, "v@:@" );
        [view_ addGestureRecognizer:rotationGestureRecognizer_];

        GestureRecognizerCocoaHandlerRegistry::instance().add( view_, this );
    }

    GestureRecognizerCocoaHandler::~GestureRecognizerCocoaHandler()
    {
        [rotationGestureRecognizer_ release];
    }

    void GestureRecognizerCocoaHandler::onRotation( GestureRecognizerHandler::RotationCallback cb )
    {
        // TODO: thread safety?
        rotationCb = cb;
    }
}

#endif

