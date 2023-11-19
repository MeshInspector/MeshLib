#pragma once

#include "MRViewerFwd.h"
#include "MRMesh/MRMeshFwd.h"
#include <mutex>
#include <queue>
#include <string>

namespace MR
{

/// queue to ignore multiple mouse moves in one frame
class MRVIEWER_CLASS ViewerEventQueue
{
public:
    // emplace event at the end of the queue
    // replace last skipable with new skipable
    MRVIEWER_API void emplace( std::string name, ViewerEventCallback cb, bool skipable = false );
    // execute all events in queue
    MRVIEWER_API void execute();
    // pop all events while they have this name
    MRVIEWER_API void popByName( const std::string& name );
    MRVIEWER_API bool empty() const;

private:
    struct NamedEvent
    {
        std::string name;
        ViewerEventCallback cb;
    };
    // important for wasm to be recursive
    mutable std::recursive_mutex mutex_;
    std::queue<NamedEvent> queue_;
    bool lastSkipable_{false};
};

} //namespace MR
