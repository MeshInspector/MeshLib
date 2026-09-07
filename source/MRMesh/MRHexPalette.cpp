#include "MRHexPalette.h"
#include "MRVector3.h"

namespace MR
{

namespace
{

constexpr std::array<Color, HexPalette::N> makeHexPalette()
{
    // for any color c: dot( c, [1,1,1] ) = 1
    constexpr Vector3f cornerColors[HexPalette::CORNER_COLORS + 1] =
    {
        { 1.0f, 0.0f, 0.0f },
        { 0.5f, 0.5f, 0.0f },
        { 0.0f, 1.0f, 0.0f },
        { 0.0f, 0.5f, 0.5f },
        { 0.0f, 0.0f, 1.0f },
        { 0.5f, 0.0f, 0.5f },
        { 1.0f, 0.0f, 0.0f }
    };
    std::array<Color, HexPalette::N> arr{};
    for ( int corner = 0; corner < HexPalette::CORNER_COLORS; ++corner )
    {
        for ( int i = 0; i < HexPalette::SIDE_COLORS; ++i )
        {
            arr[corner * HexPalette::SIDE_COLORS + i] =
                Color( lerp( cornerColors[corner], cornerColors[corner + 1], float( i ) / HexPalette::SIDE_COLORS ) );
        }
    }
    return arr;
}

} // namespace

const std::array<Color, HexPalette::N> HexPalette::colors = makeHexPalette();

} // namespace MR
