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

    /// point in body space that follows trajectory
    /// if not set body bounding box center is used
    std::optional<Vector3f> center;

    /// facing direction of body, used for initial rotation (if allowRotation)
    /// if not set body accumulative normal is used
    std::optional<Vector3f> bodyNormal;

    /// optional transform body space to trajectory space
    const AffineXf3f* b2tXf{ nullptr };

    /// if true, then body-contours will be located exactly on resulting mesh
    bool startMeshFromBody{ false };
};

/// makes mesh by moving `body` along `trajectory`
/// if allowRotation rotate it in corners
[[nodiscard]] MRMESH_API Mesh makeMovementBuildBody( const Contours3f& body, const Contours3f& trajectory,
    const MovementBuildBodyParams& params = {} );

}

