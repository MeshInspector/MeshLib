#pragma once

#include "MRMeshFwd.h"
#include "MRColor.h"

#include <vector>

namespace MR
{

/// all colors here are on cube's boundary intersected by a skew plane, which makes a hexagon;
/// gives visually distinct categorical colors for things like mesh segments or imported components
struct HexPalette
{
    /// different colors
    std::vector<Color> colors;

    /// recommended step from previous color to next color, to have big visual difference, and visit all colors in long run
    static constexpr int STEP = 17;

    MRMESH_API HexPalette();
};

} // namespace MR
