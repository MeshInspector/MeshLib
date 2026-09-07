#pragma once

#include "MRMeshFwd.h"
#include "MRSignal.h"

namespace MR
{

/// class for dispatching object tag addition/removal events
class ObjectTagEventDispatcher
{
public:
    /// returns singleton instance
    MRMESH_API static ObjectTagEventDispatcher& instance();

    using TagAddedSignal = Signal<void ( Object* obj, const std::string& tag )>;
    /// the signal is called when a tag is added to any object
    TagAddedSignal tagAddedSignal;

    using TagRemovedSignal = Signal<void ( Object* obj, const std::string& tag )>;
    /// the signal is called when a tag is removed from any object
    TagRemovedSignal tagRemovedSignal;

private:
    struct ProtectedTag {};
    explicit ObjectTagEventDispatcher( ProtectedTag );
};

} // namespace MR
