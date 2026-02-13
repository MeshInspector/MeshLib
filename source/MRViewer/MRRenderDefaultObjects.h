#pragma once
#include "MRViewer/MRRenderNameObject.h"

namespace MR
{

class VisualObject;

// Combines all the default UI `IRenderObjects`.
using RenderDefaultUiObject = RenderObjectCombinator<RenderNameObject>;

}
