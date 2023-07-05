#pragma once
#include "MRMeshFwd.h"

namespace MR
{

/// makes mesh by moving `body` along `trajectory`
/// if allowRotation rotate it in corners
[[nodiscard]] MRMESH_API Mesh makeMovementBuildBody( const Polyline3& body, const Polyline3& trajectory,
    bool allowRotation );

}

