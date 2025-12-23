#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

namespace MR
{

struct CurvePoint
{
    Vector3f pos;  ///< position on the curve
    Vector3f dir;  ///< direction along the curve
    Vector3f snorm;///< the normal of the surface where the curve is located
};

using CurveFunc = std::function<CurvePoint(float)>;

} //namespace MR
