#pragma once

#include "MRViewerFwd.h"
#include <MRMesh/MRVector3.h>

namespace MR::SpaceMouse
{

struct Parameters
{
    Vector3f translateScale{ 50.f, 50.f, 50.f }; // range [1; 100]
    Vector3f rotateScale{ 50.f, 50.f, 50.f }; // range [1; 100]

    /// it could be useful on windows if SpaceMouse driver sends mouse scroll events
    /// to avoid excessive zoom while using space mouse
    bool suppressMouseScrollZoom{ false };
};

} //namespace MR
