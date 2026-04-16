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

struct CurveFunc
{
    /// curve given as a function: position along curve -> point
    std::function<CurvePoint( float )> func;

    /// total length of the given curve
    float totalLength = 1.0f;

    /// To allow passing Python lambdas into `func`.
    MR_BIND_PREFER_UNLOCK_GIL_WHEN_USED_AS_PARAM
};

/// curve given as a number of points on it samples at arbitrary steps
using CurvePoints = std::vector<CurvePoint>;

} //namespace MR
