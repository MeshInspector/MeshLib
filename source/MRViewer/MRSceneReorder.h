#pragma once

#include "MRViewerFwd.h"
#include <vector>

namespace MR
{

struct SceneReorder
{
    /// objects that will be moved
    std::vector<Object*> who;

    /// target object
    Object* to{ nullptr };

    /// if false, each "who" will be attached to "to" as last child,
    /// if true, each "who" will be attached to "to"'s parent as a child before "to"
    bool before{ false };

    /// the name of added undo history scope
    std::string historyName{ "Reorder Scene" };
};

/// moves objects in scene as prescribed by (task), preserving world location of each object;
/// \return false if the move failed
MRVIEWER_API bool sceneReorderWithUndo( const SceneReorder & task );

/// moves all children from one object to another
/// \return false if the move failed
MRVIEWER_API bool moveAllChildrenWithUndo( Object& oldParent, Object& newParent, const std::string& historyName = "Move Children" );

} //namespace MR
