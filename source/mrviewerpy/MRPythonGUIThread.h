#pragma once
#include "MRViewer/MRCommandLoop.h"
#include "MRPython/MRPybind11.h"

namespace MR
{

// CommandLoop::runCommandFromGUIThread for a binding: waits with the GIL released, since the GUI thread
// may need the GIL to get to the command, e.g. the macOS main thread still running Python code
template<typename F>
void pythonRunCommandFromGUIThread( F&& f )
{
    pybind11::gil_scoped_release gilRelease;
    CommandLoop::runCommandFromGUIThread( std::forward<F>( f ) );
}

} // namespace MR
