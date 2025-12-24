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

/// curve given as a function: time -> point
using CurveFunc = std::function<CurvePoint(float)>;

struct CurvePointTime : CurvePoint
{
    float time = 0;
};

/// curve given as vector of points in ascending time values
using CurvePoints = std::vector<CurvePointTime>;

} //namespace MR
