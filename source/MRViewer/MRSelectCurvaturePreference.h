#pragma once
#include "exports.h"

namespace MR
{
enum class PathPreference
{
    Geodesic,
    Convex,
    Concave
};

/// draws a Combo with three PathPreference options
/// returns multiplier before dihedral angle sine in edge metric calculation (positive to prefer concave angles, negative - convex)
/// \param pp is passed by pointer because in can be changed inside
float MRVIEWER_API SelectCurvaturePreference( PathPreference* pp, float menuScaling );
}
