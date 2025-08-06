#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRVector2.h"

#include <array>
#include <optional>

namespace Json { class Value; }

namespace MR
{

/// class with CNC machine emulation settings
class MRMESH_CLASS CNCMachineSettings
{
public:
    // enumeration of axes of rotation
    enum class RotationAxisName
    {
        A,
        B,
        C
    };
    using RotationAxesOrder = std::vector<RotationAxisName>;
    using RotationLimits = std::optional<Vector2f>;

    static int getAxesCount() { return int( RotationAxisName::C ) + 1; }

    // rotationAxis length will be more then 0.01
    MRMESH_API void setRotationAxis( RotationAxisName paramName, const Vector3f& rotationAxis );
    MRMESH_API const Vector3f& getRotationAxis( RotationAxisName paramName ) const;
    // rotationLimits = {min, max}
    // valid range -180 <= min < max <= 180
    MRMESH_API void setRotationLimits( RotationAxisName paramName, const RotationLimits& rotationLimits );
    MRMESH_API const RotationLimits& getRotationLimits( RotationAxisName paramName ) const;
    // duplicated values will be removed (ABAAC - > ABC)
    MRMESH_API void setRotationOrder( const RotationAxesOrder& rotationAxesOrder );
    const RotationAxesOrder& getRotationOrder() const { return rotationAxesOrder_; }
    // set feedrate idle. valid range - [0, 100000]
    // 0 - feedrate idle set as maximum feedrate on any action, or 100 if feedrate is not set in any action
    MRMESH_API void setFeedrateIdle( float feedrateIdle );
    float getFeedrateIdle() const { return feedrateIdle_; }
    void setHomePosition( const Vector3f& homePosition ) { homePosition_ = homePosition; }
    const Vector3f& getHomePosition() const { return homePosition_; }

    MRMESH_API bool operator==( const CNCMachineSettings& rhs );
    bool operator!=( const CNCMachineSettings& rhs ) { return !(*this == rhs); }

    MRMESH_API Json::Value saveToJson() const;
    MRMESH_API bool loadFromJson( const Json::Value& jsonValue );

private:
    // direction of axes around which the rotation occurs A, B, C
    std::array<Vector3f, 3> rotationAxes_ = { Vector3f::minusX(), Vector3f::minusY(), Vector3f::plusZ() };
    // rotation limits
    std::array<RotationLimits, 3> rotationLimits_;
    // order of application of rotations
    RotationAxesOrder rotationAxesOrder_ = { RotationAxisName::A, RotationAxisName::B, RotationAxisName::C };
    // feedrate idle. 0 - feedrate idle set as maximum feedrate on any action, or 100 if feedrate is not set in any action
    float feedrateIdle_ = 10000.f;
    Vector3f homePosition_;
};

}
