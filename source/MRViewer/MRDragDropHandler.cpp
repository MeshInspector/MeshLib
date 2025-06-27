#include "MRDragDropHandler.h"
#include "MRDragDropWin32Handler.h"

namespace MR
{

std::unique_ptr<MR::IDragDropHandler> getDragDropHandler( GLFWwindow* window )
{
#ifdef _WIN32
    return std::make_unique<MR::DragDropWin32Handler>( window );
#else
    (void) window;
    return {};
#endif
}

}
