#pragma once

#include "MRMesh/MRFlagOperators.h"

namespace MR
{

/// Various passes of the 3D rendering.
enum class RenderModelPassMask
{
    Opaque = 1 << 0,
    Transparent = 1 << 1,
#ifndef MRMESH_NO_OPENVDB
    VolumeRendering = 1 << 2,
#endif
    NoDepthTest = 1 << 3,

    All =
        Opaque | Transparent | NoDepthTest
#ifndef MRMESH_NO_OPENVDB
        | VolumeRendering
#endif
};
MR_MAKE_FLAG_OPERATORS( RenderModelPassMask )

}
