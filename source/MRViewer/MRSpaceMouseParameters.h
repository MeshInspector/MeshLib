#pragma once

#include "MRViewerFwd.h"
#include <MRMesh/MRVector3.h>

namespace MR
{

struct SpaceMouseParameters
{
    Vector3f translateScale{ 50.f, 50.f, 50.f }; // range [1; 100]
    Vector3f rotateScale{ 50.f, 50.f, 50.f }; // range [1; 100]
};

} //namespace MR
