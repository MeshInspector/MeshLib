#pragma once

#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRViewportId.h"
#include "MRViewer/exports.h"
#include "MRViewer/MRImGuiMeasurementIndicators.h"

namespace MR::UI
{

// Use this in combination with `DrawSceneUiListener`.
// Remember that you don't have to allocate this on the heap, and can store this directly in your plugin per object (pass around a non-owning `shared_ptr`).
class MRVIEWER_CLASS NonOverlappingLabelTask : public BasicUiRenderTask
{
public:
    struct Params
    {
        // Mandatory:
        float menuScaling = 1;
        ViewportId viewportId;
        Vector3f worldPos;
        std::string id; // Must not repeat per frame per viewport, isn't displayed anywhere.
        std::string text;

        // Optional:
        std::shared_ptr<const ImGuiMeasurementIndicators::Params> indicatorParams;
    };

private:
    Params params_;
    ImVec2 finalScreenPos_;

public:
    MRVIEWER_API NonOverlappingLabelTask();
    MRVIEWER_API NonOverlappingLabelTask( Params params );

    MRVIEWER_API void earlyBackwardPass( const BackwardPassParams& params ) override;
    MRVIEWER_API void renderPass() override;
};

// Draws a label on top of the viewport area.
// `id` should be a unique ID persistent across frames.
MRVIEWER_API void nonOverlappingLabel();

}
