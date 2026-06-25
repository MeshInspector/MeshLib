#include "MRObjectTagEventDispatcher.h"

namespace MR
{

ObjectTagEventDispatcher& ObjectTagEventDispatcher::instance()
{
    static ObjectTagEventDispatcher sInstance{ ProtectedTag{} };
    return sInstance;
}

ObjectTagEventDispatcher::ObjectTagEventDispatcher( ProtectedTag )
{
}

} // namespace MR
