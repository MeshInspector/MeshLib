#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRExpected.h"

namespace MR
{

/// structure with parameters for `compensateRadius` function
struct CompensateRadiusParams
{
    /// Z direction of milling tool
    Vector3f direction;

    ///  radius of spherical tool
    float toolRadius{ 0.0f };

    /// size of pixel in distance map that is used for compensation
    /// negative value here means auto detect pixel size
    float pixelSize{ -1.0f };

    ProgressCallback callback;
};

/// compensate spherical milling tool radius in given mesh making it possible to mill it
/// returns new mesh with compensated radius and no undercuts in given direction
[[nodiscard]] MRMESH_API Expected<Mesh> compensateRadius( const Mesh& mesh, const CompensateRadiusParams& params );

}
