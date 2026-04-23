#pragma once

// Shared helpers for `*.Mcp.cpp` translation units. Inline-only; no new .cpp.

#ifndef MESHLIB_NO_MCP

#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRViewer.h"

namespace MR::Mcp
{

// Give the viewer a few frames to reflect the mutation before the next tool call sees stale state.
inline void skipFramesAfterInput()
{
    for ( int i = 0; i < MR::getViewerInstance().forceRedrawMinimumIncrementAfterEvents; ++i )
        MR::CommandLoop::runCommandFromGUIThread( [] {} );
}

} // namespace MR::Mcp

#endif // MESHLIB_NO_MCP
