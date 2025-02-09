#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

namespace MR
{

/// structure with parameters for `compensateRadius2` function
/*struct CompensateRadiusParams2
{
    ///  radius of spherical tool
    float toolRadius{ 0.0f };
};*/

struct SphericalMillingCutter
{
    Vector3f center;  ///< of spherical part
    float radius = 0; ///< of spherical part
};

[[nodiscard]] MRMESH_API VertBitSet findVerticesInsideTool( const Mesh& mesh, const SphericalMillingCutter& tool );

} //namespace MR
