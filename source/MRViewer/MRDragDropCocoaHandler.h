#pragma once
#ifdef __APPLE__
#include "MRDragDropHandler.h"

namespace MR
{

// macOS Drag & Drop handler. GLFW already delivers the actual file drop on macOS, but it does not
// emit drag enter/over/leave events, so the drop-zone highlight never shows. This handler injects
// the missing NSDraggingDestination methods into the GLFW content view to emit those events.
// Pure C++ header (no Objective-C) so the C++ factory can include it; the implementation lives in
// the .mm. Mirrors DragDropWin32Handler.
class DragDropCocoaHandler : public IDragDropHandler
{
public:
    explicit DragDropCocoaHandler( GLFWwindow* window );
    ~DragDropCocoaHandler() override;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}
#endif
