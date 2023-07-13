#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include <array>

namespace MR
{

struct CNCMachineSettings
{
    enum class RotationAxisName
    {
        A,
        B,
        C,
        Count
    };
    using RotationAxisOrder = std::vector<RotationAxisName>;

    std::array<Vector3f, 3> rotationAxes = { Vector3f::minusX(), Vector3f::minusY(), Vector3f::plusZ() };
    RotationAxisOrder rotationAxesOrder = { RotationAxisName::A, RotationAxisName::B, RotationAxisName::C };
};

}
