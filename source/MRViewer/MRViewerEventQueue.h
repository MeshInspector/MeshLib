#pragma once

#include "MRViewerNamedEvent.h"
#include "MRMesh/MRMeshFwd.h"
#include <mutex>
#include <queue>

namespace MR
{

/// queue to ignore multiple mouse moves in one frame
class MRVIEWER_CLASS ViewerEventQueue
{
public:
    // emplace event at the end of the queue
    // replace last skipable with new skipable
    MRVIEWER_API void emplace( ViewerNamedEvent event, bool skipable = false );
    // execute all events in queue
    MRVIEWER_API void execute();
    // pop all events while they have this name
    MRVIEWER_API void popByName( const std::string& name );
    MRVIEWER_API bool empty() const;

private:
    // important for wasm to be recursive
    mutable std::recursive_mutex mutex_;
    std::queue<ViewerNamedEvent> queue_;
    bool lastSkipable_{false};
};

} //namespace MR
