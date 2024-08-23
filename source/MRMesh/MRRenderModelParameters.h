#pragma once

#include "MRMesh/MRFlagOperators.h"

namespace MR
{

/// Various passes of the 3D rendering.
enum class RenderModelPassMask
{
    Opaque = 1 << 0,
    Transparent = 1 << 1,
    VolumeRendering = 1 << 2,
    NoDepthTest = 1 << 3,

    All =
        Opaque | Transparent | NoDepthTest | VolumeRendering
};
MR_MAKE_FLAG_OPERATORS( RenderModelPassMask )

}
