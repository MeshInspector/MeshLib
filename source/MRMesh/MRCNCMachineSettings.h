#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include <array>

namespace MR
{

/// class with CNC machine emulation settings
class CNCMachineSettings
{
public:
    // enumeration of axes of rotation
    enum class RotationAxisName
    {
        A,
        B,
        C,
        Count
    };
    using RotationAxesOrder = std::vector<RotationAxisName>;

    MRMESH_API void setRotationParams( RotationAxisName paramName, const Vector3f& rotationAxis );
    MRMESH_API Vector3f getRotationParams( RotationAxisName paramName ) const;
    MRMESH_API void setRotationOrder( const RotationAxesOrder& rotationAxesOrder );
    const RotationAxesOrder& getRotationOrder() const { return rotationAxesOrder_; }
private:
    // direction of axes around which the rotation occurs A, B, C
    std::array<Vector3f, 3> rotationAxes_ = { Vector3f::minusX(), Vector3f::minusY(), Vector3f::plusZ() };
    // order of application of rotations
    RotationAxesOrder rotationAxesOrder_ = { RotationAxisName::A, RotationAxisName::B, RotationAxisName::C };
};

}
