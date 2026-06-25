#pragma once

#include "MRViewerFwd.h"
#include <MRMesh/MRBox.h>

namespace MR
{

// Fit mode ( types of objects for which the fit is applied )
enum class FitMode
{
    Visible, // fit all visible objects
    SelectedPrimitives, // fit only selected primitives
    SelectedObjects, // fit only selected objects
    SelectableObjects, // fit only selectable objects (exclude ancillary objects)
    CustomObjectsList // fit only given objects (need additional objects list)
};

struct BaseFitParams
{
    float factor{ 1.f }; // part of the screen for scene location
    // snapView - to snap camera angle to closest canonical quaternion
    // orthographic view: camera moves a bit, fit FOV by the whole width or height
    // perspective view: camera is static, fit FOV to closest border.
    bool snapView{ false };
};

struct FitDataParams : BaseFitParams
{
    FitMode mode{ FitMode::Visible }; // fit mode
    std::vector<std::shared_ptr<VisualObject>> objsList; // custom objects list. used only with CustomObjectsList mode

    FitDataParams( float factor_ = 1.f, bool snapView_ = false, FitMode mode_ = FitMode::Visible,
        const std::vector<std::shared_ptr<VisualObject>>& objsList_ = {} ) :
        BaseFitParams{ factor_, snapView_ },
        mode( mode_ ),
        objsList( objsList_ )
    {};
};

struct FitBoxParams : BaseFitParams
{
    Box3f worldBox; // box in world space to fit

    FitBoxParams( const Box3f& worldBox_, float factor_ = 1.f, bool snapView_ = false ) :
        BaseFitParams{ factor_, snapView_ },
        worldBox( worldBox_ )
    {};
};

} //namespace MR
