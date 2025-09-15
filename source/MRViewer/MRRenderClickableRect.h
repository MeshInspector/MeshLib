#pragma once

#include "MRMesh/MRIRenderObject.h"
#include "MRViewer/exports.h"

#include "MRViewer/MRImGui.h"

namespace MR
{

// This is a helper base class for writing `BasicUiRenderTask`s that let you click a rectangle in the scene.
// Note, this task relies on preserving state between the frames.
class MRVIEWER_CLASS BasicClickableRectUiRenderTask : public BasicUiRenderTask
{
public:
    BasicClickableRectUiRenderTask() = default;

    // No-op the assignments, because we want to preserve the state across frames.
    BasicClickableRectUiRenderTask( const BasicClickableRectUiRenderTask& ) {}
    BasicClickableRectUiRenderTask& operator=( const BasicClickableRectUiRenderTask& ) { return *this; }

    virtual ~BasicClickableRectUiRenderTask() = default;

    // This is called when the click happens.
    virtual void onClick() = 0;

    // This is what ultimately calls `onClick()` if the certain conditions hold.
    MRVIEWER_API void earlyBackwardPass( const BackwardPassParams& backParams ) override;

    // Set those to set the clickable area. Zero both to disable the clicks.
    ImVec2 clickableCornerA_;
    ImVec2 clickableCornerB_;

    // Read these to decide how to render.
    bool isHovered = false;
    bool isActive = false;
};

}
