#pragma once

#include "MRViewer/MRRenderMeasurementsObject.h"
#include "MRViewer/MRRenderNameObject.h"

namespace MR
{

// Combines all the default UI `IRenderObjects`.
using RenderDefaultUiObject = RenderObjectCombinator<RenderNameObject, RenderMeasurementsObject>;

}
