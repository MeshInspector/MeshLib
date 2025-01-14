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

    /// region of the mesh that will be compensated
    /// it should not contain closed components
    /// it is updated during algorithm
    /// also please note that boundaries of the region are fixed
    FaceBitSet* region{ nullptr };

    ProgressCallback callback;
};

/// compensate spherical milling tool radius in given mesh region making it possible to mill it
/// please note that it will change topology inside region
/// also please note that boundaries of the region are fixed
[[nodiscard]] MRMESH_API Expected<void> compensateRadius( Mesh& mesh, const CompensateRadiusParams& params );

}
