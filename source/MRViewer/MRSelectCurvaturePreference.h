#pragma once
#include "exports.h"

namespace MR
{
enum class CurvaturePreferenceMode
{
    Geodesic,
    Convex,
    Concave
};

float MRVIEWER_API SelectCurvaturePreference( CurvaturePreferenceMode* cp, float menuScaling );
}
