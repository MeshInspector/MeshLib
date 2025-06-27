#pragma once
#ifdef _WIN32
#include "MRDragDropHandler.h"
#include <windef.h>

namespace MR
{

class WinDropTarget;

class DragDropWin32Handler : public IDragDropHandler
{
public:
    DragDropWin32Handler( GLFWwindow* window );
    ~DragDropWin32Handler();
private:
    HWND window_{ nullptr };
    std::unique_ptr<WinDropTarget> winDropTartget_;
};

}
#endif
