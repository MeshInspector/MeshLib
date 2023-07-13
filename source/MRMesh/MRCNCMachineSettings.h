#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include <array>

namespace MR
{

/// structure with cnc machine emulation settings
struct CNCMachineSettings
{
    // enumeration of axes of rotation
    enum class RotationAxisName
    {
        A,
        B,
        C,
        Count
    };
    using RotationAxisOrder = std::vector<RotationAxisName>;

    // direction of axes around which the rotation occurs A, B, C
    std::array<Vector3f, 3> rotationAxes = { Vector3f::minusX(), Vector3f::minusY(), Vector3f::plusZ() };
    // order of application of rotations
    RotationAxisOrder rotationAxesOrder = { RotationAxisName::A, RotationAxisName::B, RotationAxisName::C };
};

}
