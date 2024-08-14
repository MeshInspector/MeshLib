#pragma once

#include "MRViewer/MRRenderDefaultObjects.h"
#include "MRViewer/MRUINonOverlappingLabels.h"

namespace MR
{

class ObjectImGuiLabel;

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
