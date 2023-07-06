#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include <optional>

namespace MR
{

struct MovementBuildBodyParams
{
    /// if this flag is set, rotate body in trajectory vertices
    /// according to its rotation
    /// otherwise body movement will be done without any rotation
    bool allowRotation{ true };
    /// point around which body is rotated (if allowRotation)
    /// if not set body bounding box center is used
    std::optional<Vector3f> rotationCenter;
    /// facing direction of body, used for initial rotation (if allowRotation)
    /// if not set body accumulative normal is used
    std::optional<Vector3f> bodyNormal;
    /// optional transform trajectory space to body space
    const AffineXf3f* t2bXf{ nullptr };
};

/// makes mesh by moving `body` along `trajectory`
/// if allowRotation rotate it in corners
[[nodiscard]] MRMESH_API Mesh makeMovementBuildBody( const Polyline3& body, const Polyline3& trajectory,
    const MovementBuildBodyParams& params = {} );

}

