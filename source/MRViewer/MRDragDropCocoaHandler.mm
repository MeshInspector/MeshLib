#ifdef __APPLE__

#include "MRDragDropCocoaHandler.h"
#include "MRViewer.h"
#include "MRViewerSignals.h"

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

#include <AppKit/AppKit.h>

#include <objc/objc-runtime.h>

#include <cmath>
#include <map>

namespace
{

// Maps the GLFW content view to the handler that injected the dragging methods, so the static
// callbacks (installed on the shared content-view class) can find their owner and skip firing
// after the handler has been destroyed.
class DragDropCocoaHandlerRegistry
{
public:
    static DragDropCocoaHandlerRegistry& instance()
    {
        static DragDropCocoaHandlerRegistry instance;
        return instance;
    }

    void add( NSView* view, MR::DragDropCocoaHandler* handler )
    {
        registry_.emplace( view, handler );
    }

    void remove( NSView* view )
    {
        registry_.erase( view );
    }

    [[nodiscard]] MR::DragDropCocoaHandler* find( NSView* view ) const
    {
        const auto it = registry_.find( view );
        if ( it != registry_.end() )
            return it->second;
        else
            return nullptr;
    }

private:
    std::map<NSView*, MR::DragDropCocoaHandler*> registry_;
};

}

namespace MR
{

class DragDropCocoaHandler::Impl
{
public:
    Impl( GLFWwindow* window, DragDropCocoaHandler* handler );
    ~Impl();

    static NSDragOperation onDraggingEntered( NSView* view, SEL cmd, id<NSDraggingInfo> sender );
    static NSDragOperation onDraggingUpdated( NSView* view, SEL cmd, id<NSDraggingInfo> sender );
    static void onDraggingExited( NSView* view, SEL cmd, id<NSDraggingInfo> sender );

private:
    // convert the Cocoa dragging location to the Win32/Wasm convention and emit dragOverSignal
    static void emitDragOver_( NSView* view, id<NSDraggingInfo> sender );

    NSView* view_;
    IMP previousDraggingEnteredMethod_ = nil;
};

DragDropCocoaHandler::DragDropCocoaHandler( GLFWwindow* window )
    : impl_( std::make_unique<Impl>( window, this ) )
{
}

DragDropCocoaHandler::~DragDropCocoaHandler() = default;

DragDropCocoaHandler::Impl::Impl( GLFWwindow* window, DragDropCocoaHandler* handler )
    : view_( ( (NSWindow*)glfwGetCocoaWindow( window ) ).contentView )
{
    Class cls = [view_ class];

    // GLFW implements draggingEntered: (it returns NSDragOperationGeneric) and performDragOperation:
    // (it delivers the actual drop). Chain to GLFW's draggingEntered: so its return value is kept,
    // and additionally emit the enter signal. performDragOperation: is left untouched so the existing
    // GLFW drop path keeps firing dragDropSignal.
    Method enteredMethod = class_getInstanceMethod( cls, @selector( draggingEntered: ) );
    const char* enteredTypes = enteredMethod ? method_getTypeEncoding( enteredMethod ) : "L@:@";
    if ( enteredMethod )
        previousDraggingEnteredMethod_ = method_setImplementation( enteredMethod, (IMP)Impl::onDraggingEntered );
    else
        class_addMethod( cls, @selector( draggingEntered: ), (IMP)Impl::onDraggingEntered, enteredTypes );

    // GLFW does not implement draggingUpdated:/draggingExited:; install ours (replace if a future
    // GLFW ever adds them). draggingUpdated: shares draggingEntered:'s signature.
    if ( !class_addMethod( cls, @selector( draggingUpdated: ), (IMP)Impl::onDraggingUpdated, enteredTypes ) )
        class_replaceMethod( cls, @selector( draggingUpdated: ), (IMP)Impl::onDraggingUpdated, enteredTypes );
    if ( !class_addMethod( cls, @selector( draggingExited: ), (IMP)Impl::onDraggingExited, "v@:@" ) )
        class_replaceMethod( cls, @selector( draggingExited: ), (IMP)Impl::onDraggingExited, "v@:@" );

    DragDropCocoaHandlerRegistry::instance().add( view_, handler );
}

DragDropCocoaHandler::Impl::~Impl()
{
    if ( previousDraggingEnteredMethod_ != nil )
    {
        Method enteredMethod = class_getInstanceMethod( [view_ class], @selector( draggingEntered: ) );
        if ( enteredMethod )
            method_setImplementation( enteredMethod, previousDraggingEnteredMethod_ );
    }
    // draggingUpdated:/draggingExited: stay installed (Objective-C methods cannot be removed); the
    // registry lookup below makes them no-ops once this handler is gone.
    DragDropCocoaHandlerRegistry::instance().remove( view_ );
}

NSDragOperation DragDropCocoaHandler::Impl::onDraggingEntered( NSView* view, SEL cmd, id<NSDraggingInfo> sender )
{
    NSDragOperation op = NSDragOperationGeneric;
    auto* handler = DragDropCocoaHandlerRegistry::instance().find( view );
    if ( !handler )
        return op;

    if ( handler->impl_->previousDraggingEnteredMethod_ != nil )
    {
        using EnteredFn = NSDragOperation (*)( id, SEL, id );
        op = ( (EnteredFn)handler->impl_->previousDraggingEnteredMethod_ )( view, cmd, sender );
    }

    auto& v = getViewerInstance();
    v.emplaceEvent( "Drag enter", [&v] ()
    {
        v.signals().dragEntranceSignal( true );
    } );
    v.postEmptyEvent();

    return op;
}

NSDragOperation DragDropCocoaHandler::Impl::onDraggingUpdated( NSView* view, SEL, id<NSDraggingInfo> sender )
{
    emitDragOver_( view, sender );
    // match GLFW's draggingEntered:, which accepts any drag with NSDragOperationGeneric
    return NSDragOperationGeneric;
}

void DragDropCocoaHandler::Impl::onDraggingExited( NSView* view, SEL, id<NSDraggingInfo> )
{
    if ( !DragDropCocoaHandlerRegistry::instance().find( view ) )
        return;

    auto& v = getViewerInstance();
    v.emplaceEvent( "Drag leave", [&v] ()
    {
        v.signals().dragEntranceSignal( false );
    } );
    v.postEmptyEvent();
}

void DragDropCocoaHandler::Impl::emitDragOver_( NSView* view, id<NSDraggingInfo> sender )
{
    if ( !DragDropCocoaHandlerRegistry::instance().find( view ) )
        return;

    const NSPoint windowPoint = [sender draggingLocation];
    const NSPoint viewPoint = [view convertPoint:windowPoint fromView:nil];
    // Cocoa coordinates have origin at bottom-left; flip Y to the top-left origin expected by the
    // Win32/Wasm handlers and the highlight renderer. Logical units are scaled to framebuffer
    // pixels by pixelRatio inside the queued event, matching the other platforms.
    const double x = viewPoint.x;
    const double y = view.bounds.size.height - viewPoint.y;

    auto& v = getViewerInstance();
    v.emplaceEvent( "Drag over", [&v, x, y] ()
    {
        v.signals().dragOverSignal( int( std::round( x * v.pixelRatio ) ), int( std::round( y * v.pixelRatio ) ) );
    }, true );
    v.postEmptyEvent();
}

}

#endif
