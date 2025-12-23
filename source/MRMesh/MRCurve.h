#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

namespace MR
{

struct CurvePoint
{
    Vector3f pos;
    Vector3f dir;  // along curve
    Vector3f norm; // upward from the surface
};

using CurveFunc = std::function<CurvePoint(float)>;

} //namespace MR
