#pragma once

#include "MRViewer/MRRenderNameObject.h"

namespace MR
{

// Combines all the default UI `IRenderObjects`, plus your own render objects.
template <typename ...ExtraBases>
using RenderDefaultUiObject = RenderObjectCombinator<RenderNameObject, ExtraBases...>;

}
