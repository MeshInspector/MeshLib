#pragma once

#include "MRViewer/MRRenderDefaultObjects.h"
#include "MRViewer/MRUINonOverlappingLabels.h"

namespace MR
{

class ObjectImGuiLabel;

// The implementation of `IRenderObject` used by `ObjectImGuiLabel`. Draws an label using ImGui.
class RenderImGuiLabelObject : public RenderObjectCombinator<RenderDefaultUiObject, RenderResetDirtyComponent>
{
    const ObjectImGuiLabel* object_ = nullptr;
    UI::NonOverlappingLabelTask task_;
    ImGuiMeasurementIndicators::Params indicatorParams_;
public:
    MRVIEWER_API RenderImGuiLabelObject( const VisualObject& object );
    MRVIEWER_API void renderUi( const UiRenderParams& params ) override;
};

}
