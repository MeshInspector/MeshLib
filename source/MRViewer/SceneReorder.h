#pragma once

#include "MRViewerFwd.h"
#include <vector>

namespace MR
{

struct SceneReorder
{
    /// object that will be moved
    std::vector<Object*> who;

    /// address object
    Object* to{ nullptr };

    /// if false "who" will be attached to "to" as last child, otherwise "who" will be attached to "to"'s parent as child before "to"
    bool before{ false };
};

/// moves objects in scene as prescribed by (task), preserving world location of each object;
/// \return false if the move failed
bool sceneReorderWithUndo( const SceneReorder & task );

} //namespace MR
