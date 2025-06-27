#pragma once
#include "MRViewerFwd.h"

struct GLFWwindow;

namespace MR
{

// this class is needed to emit detailed Drag & Drop events to Viewer on different platforms
class IDragDropHandler
{
public:
    virtual ~IDragDropHandler() = default;
};

// returns platform specific Drag & Drop handler that will emit drag & drop events to Viewer
// null means that this platform does not support detailed Drag & Drop events yet
std::unique_ptr<IDragDropHandler> getDragDropHandler( GLFWwindow* window );

}