#pragma once

#include "MRMeshFwd.h"
#include "MRColor.h"

#include <array>

namespace MR
{

/// all colors here are on cube's boundary intersected by a skew plane, which makes a hexagon;
/// gives visually distinct categorical colors for things like mesh segments or imported components
struct HexPalette
{
    static constexpr int CORNER_COLORS = 6;
    static constexpr int SIDE_COLORS = 5; // num colors between two corner colors + 1
    static constexpr int N = CORNER_COLORS * SIDE_COLORS;

    /// recommended step from previous color to next color, to have big visual difference, and visit all colors in long run
    static constexpr int STEP = 17;

    /// the palette colors, populated at compile time
    MRMESH_API static const std::array<Color, N> colors;

    /// returns the i-th color in stride order, so successive i values give visually distinct colors
    static Color colorAtStep( int i ) { return colors[( i * STEP ) % N]; }
};

} // namespace MR
